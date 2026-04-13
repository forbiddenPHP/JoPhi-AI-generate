import imageio
import cv2
from PIL import Image
import os
import tempfile


def video_to_gif(
    input_path,
    output_path="output.gif",
    target_size_mb=10,
    fps=10,
    speed=2,
    sample_rate=10,
    max_frames=150,
    init_scale=0.5,
):
    reader = imageio.get_reader(input_path)

    frames = []
    for i, frame in enumerate(reader):
        if i % sample_rate != 0:
            continue

        frames.append(frame)
        if len(frames) >= max_frames:
            break

    print(f"Sampled {len(frames)} frames")

    scale = init_scale

    for attempt in range(6):  # 最多尝试6次压缩
        resized_frames = []

        for frame in frames:
            h, w = frame.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(frame, new_size)
            resized_frames.append(resized)

        # 转成PIL并减少颜色
        pil_frames = []
        for f in resized_frames:
            img = Image.fromarray(f)
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            pil_frames.append(img)

        # 临时保存
        tmp_path = tempfile.mktemp(suffix=".gif")
        pil_frames[0].save(
            tmp_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / (fps * speed)),
            loop=0,
        )

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"Attempt {attempt}: scale={scale:.3f}, size={size_mb:.2f}MB")

        if size_mb <= target_size_mb:
            os.rename(tmp_path, output_path)
            print(f"✅ Saved to {output_path} ({size_mb:.2f}MB)")
            return

        # 太大 → 继续缩小
        scale *= 0.7

    # 如果还不行，最后强制保存
    os.rename(tmp_path, output_path)
    print(f"⚠️ Final saved (still large): {size_mb:.2f}MB")

video_to_gif(
    "/Users/xiaoyicheng/Desktop/eccv/assets/application/camera/video.mp4",
    "assets/application/camera/video.gif",
    target_size_mb=10
)