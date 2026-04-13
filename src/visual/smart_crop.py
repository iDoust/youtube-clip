"""
Smart Crop — PySceneDetect + Face-Based Cropping
Uses real camera cut detection (not per-second sampling) for seamless transitions.

Flow:
  1. PySceneDetect → find real scene boundaries (camera cuts)
  2. Per-scene face detection → determine Scenario A or B
  3. FFmpeg crop per scene → split/merge or face-anchored
  4. Concatenate at real cut points → seamless transitions

Scenario A: 1 person → face-anchored center crop (9:16)
Scenario B: 2 persons → split & merge sides (discard center gap)
"""
import subprocess as sp
from pathlib import Path

import cv2
from rich.console import Console
from src.config import config

console = Console()

YUNET_MODEL = Path(__file__).parent.parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def _get_ffmpeg() -> str:
    """Returns the path to the bundled FFmpeg binary, or 'ffmpeg' if not found."""
    ffmpeg_bin = config.BASE_DIR / "bin" / "ffmpeg.exe"
    return str(ffmpeg_bin) if ffmpeg_bin.exists() else "ffmpeg"


class SmartCrop:
    def __init__(self):
        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model not found at {YUNET_MODEL}")

    def _detect_faces(self, frame, frame_w, frame_h, min_conf=0.80, min_size=45):
        """Detect faces using YuNet DNN. Returns list of (x, y, w, h).
        Filters out small faces that are likely photos/posters in background."""
        detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (frame_w, frame_h), min_conf)
        _, faces = detector.detect(frame)
        if faces is None:
            return []
        # Basic size filter
        raw = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]))
               for f in faces if int(f[2]) >= min_size and int(f[3]) >= min_size]
        if len(raw) <= 1:
            return raw
        # Relative size filter: discard faces < 40% of the largest face area
        # This removes phone screens, posters, photos detected as faces
        areas = [w * h for (_, _, w, h) in raw]
        max_area = max(areas)
        return [face for face, area in zip(raw, areas) if area >= max_area * 0.40]

    def _detect_scenes(self, input_path):
        """Use PySceneDetect to find real camera cut boundaries.
        Returns list of (start_sec, end_sec) tuples."""
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(input_path)
        scene_manager = SceneManager()
        # threshold=27 is default; lower = more sensitive to cuts
        scene_manager.add_detector(ContentDetector(threshold=27))
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            # No cuts detected → entire video is one scene
            duration = video.duration.get_seconds()
            return [(0.0, duration)]

        scenes = []
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append((start, end))

        return scenes

    def process(self, input_path: str, output_path: str) -> str:
        """
        Scene-boundary-based cropping.
        1. Detect real camera cuts with PySceneDetect
        2. Analyze faces in each scene (sample middle frame)
        3. Assign Scenario A or B per scene
        4. Crop each scene with FFmpeg
        5. Concatenate at real cut points (seamless)
        """
        cap = cv2.VideoCapture(input_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        if total_frames <= 0 or fps <= 0:
            return self._fallback_center_crop(input_path, output_path, frame_w, frame_h)

        # --- Phase 1: Detect real scene boundaries ---
        console.print("[cyan][*] Detecting scene boundaries (PySceneDetect)...[/cyan]")
        scenes = self._detect_scenes(input_path)
        console.print(f"[cyan]    Found {len(scenes)} scenes[/cyan]")

        # --- Phase 2: Analyze faces per scene ---
        cap = cv2.VideoCapture(input_path)
        scene_data = []  # list of (start, end, scenario, faces)

        for start, end in scenes:
            mid_sec = (start + end) / 2
            mid_frame_idx = int(mid_sec * fps)
            mid_frame_idx = min(mid_frame_idx, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            if not ret:
                scene_data.append((start, end, "A", []))
                continue

            faces = self._detect_faces(frame, frame_w, frame_h)

            # Also sample 1-2 more frames for reliability
            extra_faces = []
            for offset in [0.25, 0.75]:
                sample_sec = start + (end - start) * offset
                sample_idx = int(sample_sec * fps)
                sample_idx = min(sample_idx, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
                ret2, frame2 = cap.read()
                if ret2:
                    extra = self._detect_faces(frame2, frame_w, frame_h)
                    extra_faces.append(len(extra))

            # Majority vote: if 2 out of 3 samples have 2+ faces → Scenario B
            face_counts = [len(faces)] + extra_faces
            multi_count = sum(1 for c in face_counts if c >= 2)

            if multi_count >= 2:
                scenario = "B"
            else:
                scenario = "A"

            scene_data.append((start, end, scenario, faces))
            dur = end - start
            console.print(f"    Scene {len(scene_data)}: {start:.1f}s–{end:.1f}s ({dur:.1f}s) → {scenario} ({len(faces)} faces)")

        cap.release()

        if not scene_data:
            return self._fallback_center_crop(input_path, output_path, frame_w, frame_h)

        # --- Phase 3: Summary ---
        a_count = sum(1 for s in scene_data if s[2] == "A")
        b_count = sum(1 for s in scene_data if s[2] == "B")
        console.print(f"[cyan][*] Scene crop: {len(scene_data)} scenes ({a_count}×A, {b_count}×B)[/cyan]")

        # --- Phase 4: If only 1 scene, simple crop ---
        if len(scene_data) == 1:
            seg = scene_data[0]
            if seg[2] == "B":
                return self._crop_scenario_b(input_path, output_path, seg[3], frame_w, frame_h)
            else:
                return self._crop_scenario_a(input_path, output_path, seg[3], frame_w, frame_h)

        # --- Phase 5: Multi-scene crop ---
        return self._crop_multi_scene(input_path, output_path, scene_data, frame_w, frame_h)

    def _crop_multi_scene(self, input_path, output_path, scene_data, frame_w, frame_h):
        """Crop each scene separately, then concatenate at real cut points."""
        ffmpeg = _get_ffmpeg()
        temp_dir = Path(input_path).parent
        segment_files = []

        target_w = int(frame_h * 9 / 16)
        target_w = target_w if target_w % 2 == 0 else target_w - 1

        for idx, (start, end, scenario, faces) in enumerate(scene_data):
            seg_out = temp_dir / f"_scene_{idx}.mp4"
            seg_duration = end - start

            if seg_duration < 0.1:
                continue

            if scenario == "B" and faces:
                crop_filter = self._build_scenario_b_filter(faces, frame_w, frame_h, target_w)
                if crop_filter:
                    cmd = [
                        ffmpeg, '-y', '-i', input_path,
                        '-ss', f'{start:.3f}', '-t', f'{seg_duration:.3f}',
                        '-filter_complex', crop_filter,
                        '-map', '[out]', '-map', '0:a',
                        '-c:a', 'copy',
                        str(seg_out)
                    ]
                else:
                    crop_x = self._calc_scenario_a_x(faces, frame_w, target_w)
                    cmd = [
                        ffmpeg, '-y', '-i', input_path,
                        '-ss', f'{start:.3f}', '-t', f'{seg_duration:.3f}',
                        '-vf', f'crop={target_w}:{frame_h}:{crop_x}:0',
                        '-map', '0:v', '-map', '0:a', '-c:a', 'copy',
                        str(seg_out)
                    ]
            else:
                crop_x = self._calc_scenario_a_x(faces, frame_w, target_w)
                cmd = [
                    ffmpeg, '-y', '-i', input_path,
                    '-ss', f'{start:.3f}', '-t', f'{seg_duration:.3f}',
                    '-vf', f'crop={target_w}:{frame_h}:{crop_x}:0',
                    '-map', '0:v', '-map', '0:a', '-c:a', 'copy',
                    str(seg_out)
                ]

            res = sp.run(cmd, capture_output=True, text=True, timeout=60)
            if res.returncode != 0:
                console.print(f"[red][-] Scene {idx} failed: {res.stderr[:200]}[/red]")
                continue
            if seg_out.exists() and seg_out.stat().st_size > 0:
                segment_files.append(seg_out)

        if not segment_files:
            return self._fallback_center_crop(input_path, output_path, frame_w, frame_h)

        if len(segment_files) == 1:
            segment_files[0].rename(output_path)
            return output_path

        # Concatenate all scenes (at real cut points → seamless)
        concat_list = temp_dir / "_concat.txt"
        with open(concat_list, 'w') as f:
            for sf in segment_files:
                f.write(f"file '{sf.resolve()}'\n")

        cmd = [
            ffmpeg, '-y', '-f', 'concat', '-safe', '0',
            '-i', str(concat_list),
            '-c', 'copy',
            output_path
        ]
        res = sp.run(cmd, capture_output=True, text=True, timeout=60)

        # Cleanup
        for sf in segment_files:
            sf.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

        if res.returncode != 0:
            console.print(f"[red][-] Concat failed: {res.stderr[:200]}[/red]")
            return None

        return output_path

    def _calc_scenario_a_x(self, faces, frame_w, crop_w):
        """Calculate crop X for Scenario A (face-anchored or center)."""
        if faces:
            avg_x = sum(f[0] + f[2] // 2 for f in faces) / len(faces)
            crop_x = int(avg_x - crop_w / 2)
            return max(0, min(crop_x, frame_w - crop_w))
        return (frame_w - crop_w) // 2

    def _build_scenario_b_filter(self, faces, frame_w, frame_h, target_w):
        """Build FFmpeg filter for Scenario B split+merge. Returns filter string or None."""
        face_centers_x = sorted([f[0] + f[2] // 2 for f in faces])
        if not face_centers_x:
            return None

        min_x, max_x = min(face_centers_x), max(face_centers_x)
        if (max_x - min_x) < (frame_w * 0.15):
            return None

        boundary = (min_x + max_x) / 2
        left_cluster = [x for x in face_centers_x if x < boundary]
        right_cluster = [x for x in face_centers_x if x >= boundary]

        if not left_cluster or not right_cluster:
            return None

        left_cx = sum(left_cluster) / len(left_cluster)
        right_cx = sum(right_cluster) / len(right_cluster)
        half_w = target_w // 2

        left_x = int(left_cx - half_w / 2)
        left_x = max(0, min(left_x, int(boundary - half_w)))

        right_x = int(right_cx - half_w / 2)
        right_x = min(frame_w - half_w, max(right_x, int(boundary)))

        console.print(f"      B-crop: left_x={left_x} right_x={right_x} (faces: L={left_cx:.0f} R={right_cx:.0f})")

        return (
            f'[0:v]crop={half_w}:{frame_h}:{left_x}:0[left];'
            f'[0:v]crop={half_w}:{frame_h}:{right_x}:0[right];'
            f'[left][right]hstack[out]'
        )

    def _crop_scenario_a(self, input_path, output_path, faces, frame_w, frame_h):
        """Full-clip Scenario A crop."""
        ffmpeg = _get_ffmpeg()
        target_w = int(frame_h * 9 / 16)
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        crop_w = min(target_w, frame_w)
        crop_x = self._calc_scenario_a_x(faces, frame_w, crop_w)

        console.print(f"[cyan][*] Scenario A — Face-anchored at x={crop_x}[/cyan]")
        cmd = [
            ffmpeg, '-y', '-i', input_path,
            '-vf', f'crop={crop_w}:{frame_h}:{crop_x}:0',
            '-map', '0:v', '-map', '0:a', '-c:a', 'copy',
            output_path
        ]
        sp.run(cmd, capture_output=True, text=True, timeout=60)
        return output_path

    def _crop_scenario_b(self, input_path, output_path, faces, frame_w, frame_h):
        """Full-clip Scenario B crop."""
        ffmpeg = _get_ffmpeg()
        target_w = int(frame_h * 9 / 16)
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        crop_filter = self._build_scenario_b_filter(faces, frame_w, frame_h, target_w)

        if not crop_filter:
            return self._crop_scenario_a(input_path, output_path, faces, frame_w, frame_h)

        console.print(f"[cyan][*] Scenario B — Split & merge[/cyan]")
        cmd = [
            ffmpeg, '-y', '-i', input_path,
            '-filter_complex', crop_filter,
            '-map', '[out]', '-map', '0:a', '-c:a', 'copy',
            output_path
        ]
        res = sp.run(cmd, capture_output=True, text=True, timeout=90)
        if res.returncode != 0:
            console.print(f"[red][-] Scenario B failed: {res.stderr[:200]}[/red]")
            return None
        return output_path

    def _fallback_center_crop(self, input_path, output_path, frame_w, frame_h):
        """Dead center fallback."""
        ffmpeg = _get_ffmpeg()
        target_w = int(frame_h * 9 / 16)
        target_w = target_w if target_w % 2 == 0 else target_w - 1
        crop_x = (frame_w - target_w) // 2
        console.print("[cyan][*] Fallback — Dead center crop[/cyan]")
        cmd = [
            ffmpeg, '-y', '-i', input_path,
            '-vf', f'crop={target_w}:{frame_h}:{crop_x}:0',
            '-map', '0:v', '-map', '0:a', '-c:a', 'copy',
            output_path
        ]
        sp.run(cmd, capture_output=True, text=True, timeout=60)
        return output_path
