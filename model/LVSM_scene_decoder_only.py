# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).


import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
from utils import camera_utils, data_utils 
from .transformer import QK_Norm_TransformerBlock, init_weights
from .loss import LossComputer, LatentLossComputer
from utils import debug_utils

class Images2LatentScene(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Autoencoder
        self._init_first_stage()

        # latent space loss calc
        self.loss_latent_computer = LatentLossComputer(config) # torch.nn.MSELoss()


    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)

        return tokenizer

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, bias=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * self.config.model.first_stage_config.ddconfig.z_channels,
                bias=False,
            ),
            # nn.Sigmoid() CHECK
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)

        # Create transformer blocks
        self.transformer_blocks = [
            QK_Norm_TransformerBlock(
                config.d, config.d_head, use_qk_norm=use_qk_norm
            ) for _ in range(config.n_layer)
        ]
        
        # Apply special initialization if configured
        if config.get("special_init", False):
            for idx, block in enumerate(self.transformer_blocks):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            for block in self.transformer_blocks:
                block.apply(init_weights)
                
        self.transformer_blocks = nn.ModuleList(self.transformer_blocks)
        self.transformer_input_layernorm = nn.LayerNorm(config.d, bias=False)

    # init first stage autoencoder
    def _init_first_stage(self):
        import importlib
        cfg = self.config.model.first_stage_config

        module_path, class_name = cfg.target.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)

        model = cls(
            embed_dim=cfg.embed_dim,
            ddconfig=cfg.ddconfig,
            lossconfig=cfg.lossconfig,
        )

        # Load frozen checkpoint
        ckpt_path = cfg.params.get("ckpt_path", None)
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            # Unwrap common checkpoint wrappers
            state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=True)

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_latent_computer.eval()
        self.first_stage_model.eval()


    
    def pass_layers(self, transformer_blocks, input_tokens, gradient_checkpoint=False, checkpoint_every=1):
        """
        Helper function to pass input tokens through all transformer blocks with optional gradient checkpointing.
        
        Args:
            input_tokens: Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The input tokens to process through the transformer blocks.
            gradient_checkpoint: bool, default False
                Whether to use gradient checkpointing to save memory during training.
            checkpoint_every: int, default 1 
                Number of transformer layers to group together for gradient checkpointing.
                Only used when gradient_checkpoint=True.
                
        Returns:
            Tensor of shape [batch_size, num_views * num_patches, hidden_dim]
                The processed tokens after passing through all transformer blocks.
        """
        num_layers = len(transformer_blocks)
        
        if not gradient_checkpoint:
            # Standard forward pass through all layers
            for layer in transformer_blocks:
                input_tokens = layer(input_tokens)
            return input_tokens
            
        # Gradient checkpointing enabled - process layers in groups
        def _process_layer_group(tokens, start_idx, end_idx):
            """Helper to process a group of consecutive layers."""
            for idx in range(start_idx, end_idx):
                tokens = transformer_blocks[idx](tokens)
            return tokens
            
        # Process layer groups with gradient checkpointing
        for start_idx in range(0, num_layers, checkpoint_every):
            end_idx = min(start_idx + checkpoint_every, num_layers)
            input_tokens = torch.utils.checkpoint.checkpoint(
                _process_layer_group,
                input_tokens,
                start_idx,
                end_idx,
                use_reentrant=False
            )
            
        return input_tokens
            


    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images, pose_cond], dim=2)
    
    
    def forward(self, data_batch, has_target_image=True):

        input, target = self.process_data(data_batch, has_target_image=has_target_image, target_has_input = self.config.training.target_has_input, compute_rays=True)

        checkpoint_every = self.config.training.grad_checkpoint_every
        n_latent_vectors = self.config.model.transformer.n_latent_vectors

        # DEBUG: step counter + gating flag, computed once per forward call
        debug_cfg = self.config.get("debug", {})
        debug_enabled = debug_cfg.get("enabled", False)
        debug_interval = debug_cfg.get("interval", 500)
        self._debug_step = getattr(self, "_debug_step", 0) + 1
        do_debug = debug_enabled and (self._debug_step % debug_interval == 0)
        if do_debug:
            from utils import debug_utils
            debug_out_dir = debug_cfg.get("out_dir", "./debug_out")


        # save the pixel-space images
        input.image_pixel = input.image

        # DEBUG: raw pixel input, pre-VAE-encode
        if do_debug:
            debug_utils.dump_tensor_state(
                input.image_pixel, "01_input_pixel_raw", debug_out_dir,
                step=self._debug_step, is_latent=False
            )
            debug_utils.vae_roundtrip_test(
                self.first_stage_model, target.image, debug_out_dir, step=self._debug_step
            )



        # autoencode the images before processing
        with torch.no_grad():
            b, v, c, h, w = target.image.shape
            target.image_latent = self.first_stage_model.encode(
                target.image.reshape(b*v, c, h, w)
            ).sample().reshape(b, v, 16, h//4, w//4)

            # Need to pass the target image through the encode
            bi, vi, ci, hi, wi = input.image.shape
            input.image = self.first_stage_model.encode(
                input.image.reshape(bi*vi, ci, hi, wi)
            ).sample().reshape(bi, vi, 16, hi//4, wi//4)


        # DEBUG: post-encode latents, decoded back to check round-trip at this stage
        if do_debug:
            debug_utils.dump_tensor_state(
                target.image_latent, "02_target_latent_encoded", debug_out_dir,
                step=self._debug_step, is_latent=True, vae=self.first_stage_model
            )
            debug_utils.dump_tensor_state(
                input.image, "02_input_latent_encoded", debug_out_dir,
                step=self._debug_step, is_latent=True, vae=self.first_stage_model
            )

        # recompute rays at latent resolution (H/4, W/4)
        lh, lw = h//4, w//4
        input.ray_o, input.ray_d = self.process_data.compute_rays(
            fxfycxcy=input.fxfycxcy, c2w=input.c2w,
            h=lh, w=lw, device=input.image.device
        )
        target.ray_o, target.ray_d = self.process_data.compute_rays(
            fxfycxcy=target.fxfycxcy, c2w=target.c2w,
            h=lh, w=lw, device=input.image.device
        )

        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()

        input_img_tokens = self.image_tokenizer(posed_input_images)  # [b*v, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        
     
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]
        _, n_patches_target, _ = target_pose_tokens.size()

        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        )

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)  
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens = self.pass_layers(self.transformer_blocks, concat_img_tokens, gradient_checkpoint=True, checkpoint_every=checkpoint_every)

        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_patches, n_patches_target], dim=1
        ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches_target, d]

        # [b*v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=lh // patch_size,
            w=lw // patch_size,
            p1=patch_size,
            p2=patch_size,
            c=16
        )
        rendered_images_latent = rendered_images
        pixel_height, pixel_width = target.image_h_w

        # DEBUG: transformer's predicted latents, pre-VAE-decode -- most likely spot
        # for a scale/distribution mismatch to first show up
        if do_debug:
            debug_utils.dump_tensor_state(
                rendered_images_latent, "03_predicted_latent", debug_out_dir,
                step=self._debug_step, is_latent=True, vae=self.first_stage_model
            )
        
        # get scales to match
        bv = rendered_images.shape[0] * rendered_images.shape[1]
        rendered_images = self.first_stage_model.decode(
            rendered_images.reshape(bv, 16, rendered_images.shape[3], rendered_images.shape[4])
        ).reshape(rendered_images.shape[0], v_target, 3, pixel_height, pixel_width)
        
        rendered_images = rendered_images * 0.5 + 0.5

        # DEBUG: final decoded output, post-rescale -- compare directly against target_image_01
        if do_debug:
            debug_utils.dump_tensor_state(
                rendered_images, "04_final_render", debug_out_dir,
                step=self._debug_step, already_01=True
            )

        if has_target_image:
            target_image_01 = target.image * 0.5 + 0.5  # [0,1]

            # DEBUG: ground truth for direct visual comparison against 04_final_render
            if do_debug:
                debug_utils.dump_tensor_state(
                    target.image, "04_target_image_01", debug_out_dir,
                    step=self._debug_step, is_latent=False
                )


            loss_metrics = self.loss_latent_computer(
                rendered_images,
                target_image_01,
                rendered_images_latent,
                target.image_latent
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
            )
        
        return result



    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target

        with torch.no_grad():
            b, v, c, h, w = input.image.shape
            input.image = self.first_stage_model.encode(
                input.image.reshape(b*v, c, h, w)
            ).sample().reshape(b, v, 16, h//4, w//4)
        lh, lw = h//4, w//4
        input.ray_o, input.ray_d = self.process_data.compute_rays(
            fxfycxcy=input.fxfycxcy, c2w=input.c2w,
            h=lh, w=lw, device=input.image.device
        )

        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches_target, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches_target, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4

        video_rendering_list = []
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=n_patches* v_input)

            start_idx, end_idx = cur_chunk * n_patches_target, (cur_chunk + cur_view_chunk_size) * n_patches_target            
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches_target)

            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) # [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = self.transformer_input_layernorm(
                cur_concat_input_tokens
            )

            transformer_output_tokens = self.pass_layers(self.transformer_blocks, cur_concat_input_tokens, gradient_checkpoint=False)

            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_patches, n_patches_target], dim=1
            ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches_target, d]

            height, width = target.image_h_w

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            # [b, v_target*n_patches, p*p*3]
            video_rendering = self.image_token_decoder(pred_target_image_tokens)
            
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=lh // patch_size,
                w=lw // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=16
            )
            bv = video_rendering.shape[0] * video_rendering.shape[1]
            pixel_height, pixel_width = target.image_h_w
            video_rendering = self.first_stage_model.decode(
                video_rendering.reshape(bv, 16, video_rendering.shape[3], video_rendering.shape[4])
            ).reshape(video_rendering.shape[0], cur_view_chunk_size, 3, pixel_height, pixel_width)
            video_rendering = (video_rendering * 0.5 + 0.5).cpu()

            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering


        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["model"], strict=False)
        return 0


