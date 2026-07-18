from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def rounded_line(draw, points, fill, width):
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width // 2
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def create_master():
    size = 1024
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    dark = (29, 42, 54, 255)
    dark_inner = (36, 52, 66, 255)
    white = (245, 249, 252, 255)
    soft = (208, 220, 228, 255)
    green = (34, 158, 88, 255)

    draw.rounded_rectangle((42, 42, 982, 982), radius=205, fill=dark)
    draw.rounded_rectangle((88, 88, 936, 936), radius=165, fill=dark_inner)

    # Photo crop brackets.
    bracket_width = 58
    rounded_line(draw, [(208, 350), (208, 224), (342, 224)], white, bracket_width)
    rounded_line(draw, [(682, 224), (816, 224), (816, 350)], white, bracket_width)
    rounded_line(draw, [(208, 674), (208, 800), (342, 800)], white, bracket_width)
    rounded_line(draw, [(682, 800), (816, 800), (816, 674)], white, bracket_width)

    # Rear family members.
    draw.ellipse((274, 390, 430, 546), fill=soft)
    draw.rounded_rectangle((225, 520, 475, 760), radius=120, fill=soft)
    draw.ellipse((594, 390, 750, 546), fill=white)
    draw.rounded_rectangle((549, 520, 799, 760), radius=120, fill=white)

    # Selected central face is the optimizer focus.
    draw.rounded_rectangle((350, 504, 674, 808), radius=155, fill=green)
    draw.ellipse((404, 312, 620, 528), fill=green)
    draw.ellipse((470, 383, 494, 407), fill=white)
    draw.ellipse((530, 383, 554, 407), fill=white)
    draw.arc((469, 412, 555, 476), start=15, end=165, fill=white, width=13)

    # Small focus confirmation mark.
    rounded_line(draw, [(702, 618), (744, 660), (822, 574)], green, 30)
    return image


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    master = create_master()
    preview = master.resize((512, 512), Image.Resampling.LANCZOS)
    preview.save(ASSETS / "app_icon_preview.png")
    master.save(
        ASSETS / "app_icon.ico",
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(ASSETS / "app_icon.ico")


if __name__ == "__main__":
    main()
