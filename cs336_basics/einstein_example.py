import einops
import einx
import torch

images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
print(images.shape)

dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
print(dim_by.shape)

## Reshape
dim_value = einops.rearrange(dim_by, "d -> 1 d 1 1 1")
print(dim_value.shape)
dim_value = einx.id("d -> 1 d 1 1 1", dim_by)
print(dim_value.shape)

images_rearr = einops.rearrange(images, "b h w c -> b 1 h w c")
print(images_rearr.shape)

dimmed_images_torch = images_rearr * dim_value
print(dimmed_images_torch.shape)

dimmed_images_einsum = einops.einsum(images, dim_by, "b h w c, d -> b d h w c")
print(dimmed_images_einsum.shape)

dimmed_images_einx = einx.dot("b h w c, d -> b d h w c", images, dim_by)
print(dimmed_images_einx.shape)

torch.testing.assert_close(dimmed_images_torch, dimmed_images_einsum)
torch.testing.assert_close(dimmed_images_torch, dimmed_images_einx)


channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32 * 32, 32 * 32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(-1, channels_last.size(1) * channels_last.size(2), channels_last.size(3))
print(f"channels_last_flat.shape: {channels_last_flat.shape}")
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed_torch = channels_last_flat_transformed.view(*channels_last.shape)


channels_last_transformed_einx = einx.dot(
    "batch height_in width_in channel, (height_out width_out) (height_in width_in)"
    " -> batch height_out width_out channel",
    channels_last,
    B,
    width_in=32,
    width_out=32,
)

torch.testing.assert_close(channels_last_transformed_einx, channels_last_transformed_torch)
