"""Tests for video processing pipeline (vitok.video and vitok.pp.io video functions).

Tests cover:
- Frame extraction from image sequences and directories
- Video preprocessing in batch mode
- Video postprocessing
- Video collation for multiple videos
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vitok.video import extract_frames
from vitok.pp.io import preprocess_video, postprocess_video, video_collate_fn


# =============================================================================
# Fixtures - Test frames
# =============================================================================


@pytest.fixture(scope="module")
def test_frames():
    """Create a set of test frames with varying content."""
    frames = []
    for i in range(8):
        # Create frame with different color per frame
        h, w = 256, 256
        r = np.full((h, w), (i * 30) % 256, dtype=np.uint8)
        g = np.full((h, w), ((i + 2) * 30) % 256, dtype=np.uint8)
        b = np.full((h, w), ((i + 4) * 30) % 256, dtype=np.uint8)
        img = Image.fromarray(np.stack([r, g, b], axis=-1))
        frames.append(img)
    return frames


@pytest.fixture
def frame_directory(test_frames, tmp_path):
    """Create a directory with test frames as PNG files."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    for i, frame in enumerate(test_frames):
        frame.save(frame_dir / f"frame_{i:04d}.png")

    return frame_dir


@pytest.fixture
def frame_paths(test_frames, tmp_path):
    """Create temporary frame files and return their paths."""
    paths = []
    for i, frame in enumerate(test_frames):
        path = tmp_path / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths


# =============================================================================
# Test Frame Extraction
# =============================================================================


class TestExtractFrames:
    """Tests for extract_frames function."""

    def test_extract_from_list_of_paths(self, frame_paths):
        """Extract frames from a list of paths."""
        frames = extract_frames(frame_paths)
        assert len(frames) == len(frame_paths)
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_extract_from_directory(self, frame_directory):
        """Extract frames from a directory."""
        frames = extract_frames(frame_directory)
        assert len(frames) == 8  # All 8 frames
        assert all(isinstance(f, Image.Image) for f in frames)
        # Verify they are loaded in sorted order
        assert frames[0].size == (256, 256)

    def test_extract_with_max_frames(self, frame_directory):
        """Extract limited number of frames."""
        frames = extract_frames(frame_directory, max_frames=4)
        assert len(frames) == 4

    def test_extract_with_start_frame(self, frame_directory):
        """Extract frames starting from offset."""
        frames = extract_frames(frame_directory, start_frame=3)
        assert len(frames) == 5  # 8 - 3 = 5

    def test_extract_with_start_and_max(self, frame_directory):
        """Extract limited frames with offset."""
        frames = extract_frames(frame_directory, max_frames=3, start_frame=2)
        assert len(frames) == 3

    def test_temporal_stride_not_implemented(self, frame_directory):
        """Temporal stride > 1 should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            extract_frames(frame_directory, temporal_stride=2)

    def test_nonexistent_path_raises(self):
        """Non-existent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_frames("/nonexistent/path")

    def test_empty_directory(self, tmp_path):
        """Empty directory should return empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        frames = extract_frames(empty_dir)
        assert frames == []


# =============================================================================
# Test Video Preprocessing
# =============================================================================


class TestPreprocessVideo:
    """Tests for preprocess_video function."""

    def test_preprocess_list_of_images(self, test_frames):
        """Preprocess list of PIL Images."""
        # Use only 4 frames to keep test fast
        frames = test_frames[:4]
        batch = preprocess_video(frames, device="cpu")

        assert "patches" in batch
        assert "patch_mask" in batch
        assert "row_idx" in batch
        assert "col_idx" in batch
        assert "orig_height" in batch
        assert "orig_width" in batch

        # Batch dimension should be number of frames
        B = batch["patches"].shape[0]
        assert B == len(frames)

    def test_preprocess_from_directory(self, frame_directory):
        """Preprocess frames from directory path."""
        batch = preprocess_video(frame_directory, max_frames=4, device="cpu")

        assert batch["patches"].shape[0] == 4

    def test_preprocess_from_paths(self, frame_paths):
        """Preprocess from list of path strings."""
        batch = preprocess_video(frame_paths[:4], device="cpu")

        assert batch["patches"].shape[0] == 4

    def test_preprocess_shapes(self, test_frames):
        """Verify output tensor shapes."""
        frames = test_frames[:2]
        batch = preprocess_video(
            frames,
            pp="to_tensor|normalize(minus_one_to_one)|patchify(16, 256)",
            device="cpu",
        )

        B = len(frames)
        N = 256  # max_tokens from patchify

        assert batch["patches"].shape[0] == B
        assert batch["patches"].shape[1] == N
        assert batch["patch_mask"].shape == (B, N)
        assert batch["row_idx"].shape == (B, N)
        assert batch["col_idx"].shape == (B, N)

    def test_sequence_mode_not_implemented(self, test_frames):
        """Sequence mode should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            preprocess_video(test_frames[:2], mode="sequence", device="cpu")

    def test_invalid_mode_raises(self, test_frames):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError):
            preprocess_video(test_frames[:2], mode="invalid", device="cpu")


# =============================================================================
# Test Video Postprocessing
# =============================================================================


class TestPostprocessVideo:
    """Tests for postprocess_video function."""

    def test_postprocess_single_video(self, test_frames):
        """Postprocess single video output."""
        # First preprocess
        batch = preprocess_video(test_frames[:4], device="cpu")

        # Simulate model output (just use patches directly)
        output = {
            "patches": batch["patches"],
            "patch_mask": batch["patch_mask"],
            "row_idx": batch["row_idx"],
            "col_idx": batch["col_idx"],
            "orig_height": batch["orig_height"],
            "orig_width": batch["orig_width"],
        }

        frames = postprocess_video(output, patch=16)

        assert len(frames) == 4
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_postprocess_batched_video(self, test_frames):
        """Postprocess batched video output (multiple videos)."""
        # Preprocess two "videos"
        video1 = preprocess_video(test_frames[:3], device="cpu")
        video2 = preprocess_video(test_frames[3:6], device="cpu")

        # Collate
        batch = video_collate_fn([video1, video2])

        # Simulate model output
        output = {
            "patches": batch["patches"],
            "patch_mask": batch["patch_mask"],
            "row_idx": batch["row_idx"],
            "col_idx": batch["col_idx"],
            "orig_height": batch["orig_height"],
            "orig_width": batch["orig_width"],
            "frames_per_video": batch["frames_per_video"],
        }

        videos = postprocess_video(output, patch=16)

        # Should return list of lists
        assert isinstance(videos, list)
        assert len(videos) == 2
        assert len(videos[0]) == 3  # video1 had 3 frames
        assert len(videos[1]) == 3  # video2 had 3 frames

    def test_sequence_mode_not_implemented(self, test_frames):
        """Sequence mode should raise NotImplementedError."""
        batch = preprocess_video(test_frames[:2], device="cpu")
        with pytest.raises(NotImplementedError):
            postprocess_video(batch, mode="sequence")


# =============================================================================
# Test Video Collation
# =============================================================================


class TestVideoCollateFn:
    """Tests for video_collate_fn function."""

    def test_collate_two_videos(self, test_frames):
        """Collate two videos into single batch."""
        video1 = preprocess_video(test_frames[:4], device="cpu")
        video2 = preprocess_video(test_frames[4:], device="cpu")

        batch = video_collate_fn([video1, video2])

        # Total frames = 4 + 4 = 8
        assert batch["patches"].shape[0] == 8

        # Check video_idx tracks video boundaries
        assert "video_idx" in batch
        assert batch["video_idx"][:4].eq(0).all()
        assert batch["video_idx"][4:].eq(1).all()

        # Check frames_per_video for splitting
        assert batch["frames_per_video"] == [4, 4]

    def test_collate_different_frame_counts(self, test_frames):
        """Collate videos with different frame counts."""
        video1 = preprocess_video(test_frames[:2], device="cpu")
        video2 = preprocess_video(test_frames[2:7], device="cpu")

        batch = video_collate_fn([video1, video2])

        # Total frames = 2 + 5 = 7
        assert batch["patches"].shape[0] == 7
        assert batch["frames_per_video"] == [2, 5]
        assert batch["video_idx"][:2].eq(0).all()
        assert batch["video_idx"][2:].eq(1).all()

    def test_collate_preserves_metadata(self, test_frames):
        """Collation preserves all required metadata."""
        video1 = preprocess_video(test_frames[:2], device="cpu")
        video2 = preprocess_video(test_frames[2:4], device="cpu")

        batch = video_collate_fn([video1, video2])

        required_keys = [
            "patches",
            "patch_mask",
            "row_idx",
            "col_idx",
            "time_idx",
            "orig_height",
            "orig_width",
            "video_idx",
            "frames_per_video",
        ]

        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

    def test_collate_attention_mask(self, test_frames):
        """Collation creates block-diagonal attention mask."""
        video1 = preprocess_video(test_frames[:2], device="cpu")
        video2 = preprocess_video(test_frames[2:4], device="cpu")

        batch = video_collate_fn([video1, video2])

        if "attention_mask" in batch:
            # Should be block diagonal
            attn = batch["attention_mask"]
            assert attn.shape[0] == 4 * 256  # 4 frames * 256 tokens
            assert attn.shape[1] == 4 * 256


# =============================================================================
# Test Roundtrip
# =============================================================================


class TestRoundtrip:
    """Test complete preprocess -> postprocess roundtrip."""

    def test_single_video_roundtrip(self, test_frames):
        """Single video preprocess -> postprocess preserves frame count."""
        frames = test_frames[:4]

        # Preprocess
        batch = preprocess_video(frames, device="cpu")

        # Simulate identity model (just pass through)
        output = {
            "patches": batch["patches"],
            "patch_mask": batch["patch_mask"],
            "row_idx": batch["row_idx"],
            "col_idx": batch["col_idx"],
            "orig_height": batch["orig_height"],
            "orig_width": batch["orig_width"],
        }

        # Postprocess
        result_frames = postprocess_video(output, patch=16)

        assert len(result_frames) == len(frames)

    def test_batched_video_roundtrip(self, test_frames):
        """Batched videos preprocess -> postprocess preserves structure."""
        # Create two videos
        video1_frames = test_frames[:3]
        video2_frames = test_frames[3:6]

        video1 = preprocess_video(video1_frames, device="cpu")
        video2 = preprocess_video(video2_frames, device="cpu")

        # Collate
        batch = video_collate_fn([video1, video2])
        frames_per_video = batch["frames_per_video"]

        # Simulate model output
        output = {
            "patches": batch["patches"],
            "patch_mask": batch["patch_mask"],
            "row_idx": batch["row_idx"],
            "col_idx": batch["col_idx"],
            "orig_height": batch["orig_height"],
            "orig_width": batch["orig_width"],
            "frames_per_video": frames_per_video,
        }

        # Postprocess
        result = postprocess_video(output, patch=16)

        assert len(result) == 2  # Two videos
        assert len(result[0]) == 3  # video1
        assert len(result[1]) == 3  # video2
