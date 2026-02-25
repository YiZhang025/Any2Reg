"""
Visualization utilities for cardiac MRI registration results.

Style aligned with core.visualize_group_registration_step: one PNG (grid layout)
+ one GIF (per-frame panels). Optional MP4 export via FFmpeg.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch


def check_ffmpeg_available() -> Tuple[bool, str]:
    """
    Check if FFmpeg is available and get version info.
    
    Returns:
        (is_available, version_string)
    
    Note:
        This function will fail explicitly rather than silently if
        FFmpeg is not available, ensuring quality guarantees.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        return False, "FFmpeg found but version check failed"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, f"FFmpeg not found: {e}"


def render_frame_comparison(
    original: np.ndarray,
    warped: np.ndarray,
    template: np.ndarray,
    frame_idx: int,
    title_suffix: str = "",
) -> np.ndarray:
    """
    Render single frame comparison visualization.
    
    Args:
        original: (H, W) original frame
        warped: (H, W) warped frame
        template: (H, W) template
        frame_idx: Frame index for labeling
        title_suffix: Optional suffix for title
    
    Returns:
        RGB image array (H, W, 3) uint8
    
    Note:
        This creates a 3-panel layout: Original | Warped | Template
        For production use, additional layout options would be provided.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original Frame {frame_idx}')
    axes[0].axis('off')
    
    axes[1].imshow(warped, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Warped Frame {frame_idx}')
    axes[1].axis('off')
    
    axes[2].imshow(template, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Template{title_suffix}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Convert to RGB array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return buf


def render_all_frames(
    images: torch.Tensor,
    warped: torch.Tensor,
    template: torch.Tensor,
    method_name: str = "",
) -> List[np.ndarray]:
    """
    Render visualization for all frames.
    
    Args:
        images: (T, 1, H, W) original images
        warped: (T, 1, H, W) warped images
        template: (1, 1, H, W) template
        method_name: Method name for labeling
    
    Returns:
        List of RGB frames as numpy arrays
    
    Note:
        All frames are rendered at the same resolution and format
        to ensure consistent video encoding.
    """
    T = images.shape[0]
    frames = []
    
    # Convert to numpy
    images_np = images.squeeze(1).cpu().numpy()
    warped_np = warped.squeeze(1).cpu().numpy()
    template_np = template.squeeze().cpu().numpy()
    
    title_suffix = f" ({method_name})" if method_name else ""
    
    for t in range(T):
        frame = render_frame_comparison(
            images_np[t],
            warped_np[t],
            template_np,
            frame_idx=t,
            title_suffix=title_suffix,
        )
        frames.append(frame)
    
    return frames


def _create_grid_image(H: int, W: int, step: int = 16, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Checker/grid pattern (1,1,H,W) for overlay, matching core._create_grid_image."""
    grid = np.zeros((H, W), dtype=np.float32)
    grid[::step, :] = 1.0
    grid[:, ::step] = 1.0
    return torch.from_numpy(grid)[None, None].to(device)


def visualize_group_result(
    images: torch.Tensor,
    warped: torch.Tensor,
    flows: torch.Tensor,
    template: torch.Tensor,
    out_dir: Path,
    name: str,
    grid_step: int = 16,
) -> Tuple[Path, Path]:
    """
    One PNG + one GIF for group registration result (core.visualize_group_registration_step style).

    - PNG: 4 rows x N cols — Image | Template+grid | Warped+grid | |Warped-Template|
    - GIF: each frame = one time point: Image_i, Template+grid, Warped_i+grid, diff (RdBu).

    Args:
        images: (N, 1, H, W) original
        warped: (N, 1, H, W) warped
        flows: (N, 2, H, W) displacement
        template: (1, 1, H, W)
        out_dir: directory to save PNG and GIF
        name: base name (e.g. "any2regnet") -> {name}.png, {name}.gif
        grid_step: grid overlay step in pixels

    Returns:
        (path_png, path_gif)
    """
    from . import infer as _infer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert images.dim() == 4 and images.shape[1] == 1
    assert warped.shape == images.shape
    assert flows.shape[0] == images.shape[0] and flows.shape[1] == 2
    if template.dim() == 3:
        template = template.unsqueeze(0)
    N, _, H, W = images.shape
    device = images.device
    flows = flows.to(device)
    warped = warped.to(device)
    template = template.to(device)

    with torch.no_grad():
        imgs_np = images[:, 0].cpu().numpy()
        warped_np = warped[:, 0].cpu().numpy()
        tmpl_np = template[0, 0].cpu().numpy()
        diff_wt = warped_np - tmpl_np[None, ...]

        grid = _create_grid_image(H, W, step=grid_step, device=device)
        grid_np = grid[0, 0].cpu().numpy()
        grids_warp_np = []
        for i in range(N):
            flow_i = flows[i : i + 1].to(device)
            gi = _infer.warp_2d(grid.to(device), flow_i)
            grids_warp_np.append(gi[0, 0].cpu().numpy())
        grids_warp_np = np.stack(grids_warp_np, axis=0)

    # Limit columns for PNG if N is large (e.g. ACDC 30 frames)
    max_cols = 12
    if N > max_cols:
        indices = np.linspace(0, N - 1, max_cols, dtype=int)
    else:
        indices = np.arange(N)

    n_cols = len(indices)
    fig, axs = plt.subplots(4, n_cols, figsize=(2 * n_cols, 2.2 * 4))
    if n_cols == 1:
        axs = axs[:, None]

    for c, i in enumerate(indices):
        axs[0, c].imshow(imgs_np[i], cmap="gray")
        axs[0, c].set_title(f"Image {i}" if c == 0 else "")
        axs[0, c].axis("off")

        axs[1, c].imshow(tmpl_np, cmap="gray")
        axs[1, c].imshow(grid_np, cmap="jet", alpha=0.15)
        axs[1, c].set_title("Template" if c == 0 else "")
        axs[1, c].axis("off")

        axs[2, c].imshow(warped_np[i], cmap="gray")
        axs[2, c].imshow(grids_warp_np[i], cmap="jet", alpha=0.15)
        axs[2, c].set_title("Warped" if c == 0 else "")
        axs[2, c].axis("off")

        axs[3, c].imshow(diff_wt[i], cmap="RdBu", vmin=-0.1, vmax=0.1)
        axs[3, c].set_title("Warped−T" if c == 0 else "")
        axs[3, c].axis("off")

    plt.tight_layout()
    png_path = out_dir / f"{name}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # GIF: one frame per time point — Image_i | Template | Warped_i | diff
    gif_frames = []
    for i in range(N):
        fig_g, axs_g = plt.subplots(2, 2, figsize=(8, 8))
        axs_g[0, 0].imshow(imgs_np[i], cmap="gray")
        axs_g[0, 0].set_title(r"Image $I_i$")
        axs_g[0, 0].axis("off")
        axs_g[0, 1].imshow(tmpl_np, cmap="gray")
        axs_g[0, 1].imshow(grid_np, cmap="gray", alpha=0.3)
        axs_g[0, 1].set_title(r"Template $T$")
        axs_g[0, 1].axis("off")
        axs_g[1, 0].imshow(warped_np[i], cmap="gray")
        axs_g[1, 0].imshow(grids_warp_np[i], cmap="jet", alpha=0.15)
        axs_g[1, 0].set_title(r"Warped $I_i \circ \phi_i$")
        axs_g[1, 0].axis("off")
        axs_g[1, 1].imshow(diff_wt[i], cmap="RdBu", vmin=-0.1, vmax=0.1)
        axs_g[1, 1].set_title(r"$I_i \circ \phi_i - T$")
        axs_g[1, 1].axis("off")
        plt.tight_layout()
        fig_g.canvas.draw()
        buf = np.frombuffer(fig_g.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig_g.canvas.get_width_height()[::-1] + (3,))
        gif_frames.append(buf)
        plt.close(fig_g)

    gif_path = out_dir / f"{name}.gif"
    imageio.mimsave(str(gif_path), gif_frames, fps=4, loop=0)
    print(f"Saved {name}: {png_path.name}, {gif_path.name}")
    return png_path, gif_path


def export_mp4_strict_lossless(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
) -> None:
    """
    Export frames as strict lossless MP4.
    
    Args:
        frames: List of RGB frames (H, W, 3) uint8
        output_path: Output MP4 file path
        fps: Frames per second
    
    Raises:
        RuntimeError: If FFmpeg not available or encoding fails
    
    Note:
        Uses libx264rgb with qp=0 for mathematically lossless encoding.
        File size will be larger than compatibility mode but guarantees
        perfect pixel-level reproduction.
        Note: libx264rgb does not support -crf (use -qp 0 for lossless).

        Settings:
        - codec: libx264rgb (RGB color space, no chroma subsampling)
        - qp: 0 (lossless; -crf is not supported by libx264rgb in FFmpeg 4.x)
        - preset: veryslow (maximum compression efficiency)
        - pix_fmt: rgb24 (24-bit RGB)
    """
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        raise RuntimeError(
            f"FFmpeg required for MP4 export but not available: {ffmpeg_msg}\n"
            f"Installation instructions:\n"
            f"  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            f"  macOS: brew install ffmpeg\n"
            f"  Windows: https://ffmpeg.org/download.html"
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write frames to temporary directory (required for FFmpeg input)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Save frames as individual PNGs
        for idx, frame in enumerate(frames):
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            plt.imsave(frame_path, frame)
        
        # FFmpeg command for strict lossless
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-framerate', str(fps),
            '-i', str(tmp_path / 'frame_%04d.png'),
            '-c:v', 'libx264rgb',  # RGB codec (no color space conversion)
            '-qp', '0',  # Lossless (libx264rgb does not support -crf in FFmpeg 4.x)
            '-preset', 'veryslow',  # Maximum compression
            '-pix_fmt', 'rgb24',  # 24-bit RGB
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )
    
    print(f"Exported strict lossless MP4: {output_path}")


def export_mp4_compatibility(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 5,
    bitrate: str = "5000k",
) -> None:
    """
    Export frames as compatibility-mode MP4.
    
    Args:
        frames: List of RGB frames (H, W, 3) uint8
        output_path: Output MP4 file path
        fps: Frames per second
        bitrate: Target bitrate (e.g., "5000k")
    
    Raises:
        RuntimeError: If FFmpeg not available or encoding fails
    
    Note:
        Uses standard H.264 with yuv420p for maximum compatibility.
        This is near-lossless at high bitrate but not mathematically
        perfect due to chroma subsampling.
        
        Settings:
        - codec: libx264 (standard H.264)
        - pix_fmt: yuv420p (YUV 4:2:0, widely compatible)
        - bitrate: high (near-lossless quality)
        - preset: slow (good compression vs speed tradeoff)
    """
    ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
    if not ffmpeg_ok:
        raise RuntimeError(
            f"FFmpeg required for MP4 export but not available: {ffmpeg_msg}\n"
            f"Installation instructions:\n"
            f"  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            f"  macOS: brew install ffmpeg\n"
            f"  Windows: https://ffmpeg.org/download.html"
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        for idx, frame in enumerate(frames):
            frame_path = tmp_path / f"frame_{idx:04d}.png"
            plt.imsave(frame_path, frame)
        
        # FFmpeg command for compatibility mode
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', str(tmp_path / 'frame_%04d.png'),
            '-c:v', 'libx264',  # Standard H.264
            '-b:v', bitrate,  # High bitrate
            '-pix_fmt', 'yuv420p',  # Maximum compatibility
            '-preset', 'slow',  # Good quality/speed balance
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg encoding failed:\n"
                f"Command: {' '.join(cmd)}\n"
                f"Error: {result.stderr}"
            )
    
    print(f"Exported compatibility MP4: {output_path}")


def export_gif(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 15,
    loop: int = 0,
) -> Path:
    """
    Export frames as animated GIF (no FFmpeg required).

    Args:
        frames: List of RGB frames (H, W, 3) uint8
        output_path: Output GIF file path
        fps: Frames per second
        loop: Loop count (0 = infinite)

    Returns:
        output_path

    Note:
        Uses imageio; works in environments where FFmpeg codecs are limited.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure uint8
    out = [np.asarray(f).astype(np.uint8) for f in frames]
    imageio.mimsave(str(output_path), out, fps=fps, loop=loop)
    print(f"Exported GIF: {output_path}")
    return output_path


def export_dual_mp4(
    frames: List[np.ndarray],
    output_prefix: Path,
    fps: int = 5,
) -> Tuple[Path, Path]:
    """
    Export both lossless and compatibility MP4 versions.
    
    Args:
        frames: List of RGB frames
        output_prefix: Output path without extension (e.g., 'case_001')
        fps: Frames per second
    
    Returns:
        (lossless_path, compatibility_path)
    
    Note:
        This is the recommended export function as it provides both:
        - Lossless version for archival and analysis
        - Compatibility version for presentations and sharing
    """
    lossless_path = output_prefix.parent / f"{output_prefix.stem}_lossless.mp4"
    compat_path = output_prefix.parent / f"{output_prefix.stem}_compat.mp4"
    
    export_mp4_strict_lossless(frames, lossless_path, fps=fps)
    export_mp4_compatibility(frames, compat_path, fps=fps)
    
    return lossless_path, compat_path


def save_static_summary(
    images: torch.Tensor,
    warped: torch.Tensor,
    template: torch.Tensor,
    output_path: Path,
    method_name: str = "",
    show_grid: bool = False,
) -> None:
    """
    Save static summary figure (first, middle, last frames).
    
    Args:
        images: (T, 1, H, W) original images
        warped: (T, 1, H, W) warped images
        template: (1, 1, H, W) template
        output_path: Output PNG path
        method_name: Method name for title
        show_grid: Whether to overlay deformation grid
    
    Note:
        This provides a static alternative to video for quick inspection.
        Useful for paper figures and quick sanity checks.
    """
    T = images.shape[0]
    indices = [0, T // 2, T - 1] if T >= 3 else list(range(T))
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]
    
    images_np = images.squeeze(1).cpu().numpy()
    warped_np = warped.squeeze(1).cpu().numpy()
    template_np = template.squeeze().cpu().numpy()
    
    for row, idx in enumerate(indices):
        axes[row, 0].imshow(images_np[idx], cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Original Frame {idx}')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(warped_np[idx], cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Warped Frame {idx}')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(template_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 2].set_title('Template')
        axes[row, 2].axis('off')
    
    title = f"Registration Results: {method_name}" if method_name else "Registration Results"
    fig.suptitle(title, fontsize=14, y=0.995)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved static summary: {output_path}")
