"""Generate social card for GitHub."""

from PIL import Image, ImageDraw, ImageFont

# Configuration
WIDTH = 1280
HEIGHT = 640
BG_COLOR = "#0d1117"
TEXT_COLOR = "#ffffff"
ACCENT_COLOR = "#858585"

def make_social_card(output_path: str = "assets/social.png") -> None:
    """Generate social card image.

    Args:
        output_path: Output path for PNG.
    """
    # Create image with dark background
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Try to load DejaVuSans, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 90)
        tagline_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
        credit_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except (FileNotFoundError, OSError):
        # Fall back to default font
        title_font = ImageFont.load_default()
        tagline_font = ImageFont.load_default()
        credit_font = ImageFont.load_default()

    # Draw title centered
    title = "mcp-rag-starter"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = bbox[2] - bbox[0]
    title_x = (WIDTH - title_width) / 2
    title_y = 150

    draw.text((title_x, title_y), title, fill=TEXT_COLOR, font=title_font)

    # Draw tagline
    tagline = "Production grade RAG, packaged as an MCP server."
    bbox = draw.textbbox((0, 0), tagline, font=tagline_font)
    tagline_width = bbox[2] - bbox[0]
    tagline_x = (WIDTH - tagline_width) / 2
    tagline_y = title_y + 120

    draw.text((tagline_x, tagline_y), tagline, fill=TEXT_COLOR, font=tagline_font)

    # Draw credit at bottom right
    credit = "by @whodeanie"
    bbox = draw.textbbox((0, 0), credit, font=credit_font)
    credit_width = bbox[2] - bbox[0]
    credit_x = WIDTH - credit_width - 40
    credit_y = HEIGHT - 50

    draw.text((credit_x, credit_y), credit, fill=ACCENT_COLOR, font=credit_font)

    # Save image
    img.save(output_path)
    print(f"Generated social card: {output_path}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    output = sys.argv[1] if len(sys.argv) > 1 else "assets/social.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    make_social_card(output)
