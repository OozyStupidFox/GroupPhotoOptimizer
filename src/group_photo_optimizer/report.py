from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .scoring import image_score
from .types import ImageAnalysis, PersonTrack, ReplacementRecord


def _write_cv_image(path: Path, image: np.ndarray, quality: int = 94) -> None:
    extension = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(extension, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode image: {}".format(path))
    encoded.tofile(str(path))


def create_annotated_base(
    pixels: np.ndarray, base: ImageAnalysis, output_path: Path
) -> None:
    result = pixels.copy()
    font_scale = max(0.45, result.shape[1] / 6500.0)
    thickness = max(1, int(round(result.shape[1] / 3100.0)))
    for observation in base.observations:
        x, y, w, h = np.round(observation.bbox).astype(int)
        color = (40, 190, 40) if observation.overall_good else (30, 50, 230)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        flags = []
        if not observation.eyes_good:
            flags.append("E")
        if not observation.mouth_good:
            flags.append("M")
        label = "{}{}".format(observation.track_id or "?", ":" + "/".join(flags) if flags else "")
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(text_height + 2, y - 3)
        cv2.rectangle(
            result,
            (x, label_y - text_height - 3),
            (x + text_width + 4, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            result,
            label,
            (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    _write_cv_image(output_path, result)


def create_replacement_contact_sheet(
    before: np.ndarray,
    after: np.ndarray,
    base: ImageAnalysis,
    images: Sequence[ImageAnalysis],
    tracks: Sequence[PersonTrack],
    records: Sequence[ReplacementRecord],
    output_path: Path,
    image_loader,
) -> bool:
    replaced = [record for record in records if record.status == "replaced"]
    if not replaced:
        return False
    by_track = {item.track_id: item for item in base.observations if item.track_id}
    tracks_by_id = {track.track_id: track for track in tracks}
    cell_width, cell_height = 660, 240
    columns = 2
    rows = int(np.ceil(len(replaced) / columns))
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 245, dtype=np.uint8)
    for index, record in enumerate(replaced):
        observation = by_track.get(record.track_id)
        if observation is None:
            continue
        x, y, w, h = observation.bbox
        margin = max(w, h) * 0.55
        x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
        x1 = min(before.shape[1], int(x + w + margin))
        y1 = min(before.shape[0], int(y + h + margin))
        before_crop = before[y0:y1, x0:x1]
        after_crop = after[y0:y1, x0:x1]
        track = tracks_by_id.get(record.track_id)
        donor_observation = None if track is None else next(
            (item for item in track.observations.values() if item.image_name == record.donor_image), None
        )
        if donor_observation is None:
            continue
        donor_image = image_loader(images[donor_observation.image_index].path)
        dx, dy, dw, dh = donor_observation.bbox
        donor_margin = max(dw, dh) * 0.55
        dx0, dy0 = max(0, int(dx - donor_margin)), max(0, int(dy - donor_margin))
        dx1 = min(donor_image.shape[1], int(dx + dw + donor_margin))
        dy1 = min(donor_image.shape[0], int(dy + dh + donor_margin))
        donor_crop = donor_image[dy0:dy1, dx0:dx1]
        if before_crop.size == 0 or donor_crop.size == 0 or after_crop.size == 0:
            continue
        preview_size = (200, 195)
        previews = [
            cv2.resize(item, preview_size, interpolation=cv2.INTER_CUBIC)
            for item in (before_crop, donor_crop, after_crop)
        ]
        row, column = divmod(index, columns)
        px, py = column * cell_width, row * cell_height
        for preview_index, preview in enumerate(previews):
            start_x = px + 10 + preview_index * 215
            sheet[py + 37 : py + 232, start_x : start_x + 200] = preview
        title = "{} / {} / donor {}".format(record.track_id, record.region, record.donor_image)
        cv2.putText(sheet, title, (px + 10, py + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (25, 25, 25), 1, cv2.LINE_AA)
        for preview_index, label in enumerate(("BASE", "DONOR", "FINAL")):
            cv2.putText(sheet, label, (px + 14 + preview_index * 215, py + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    _write_cv_image(output_path, sheet, quality=96)
    return True


def write_observation_csv(path: Path, images: Sequence[ImageAnalysis]) -> None:
    columns = [
        "track_id", "image", "detection_score", "sharpness", "eye_blink",
        "eye_asymmetry", "mouth_abnormal", "expression_deviation", "eyes_good",
        "mouth_good", "quality_good", "overall_good", "x", "y", "width", "height",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for image in images:
            for item in image.observations:
                writer.writerow({
                    "track_id": item.track_id or "",
                    "image": image.path.name,
                    "detection_score": "{:.4f}".format(item.detection_score),
                    "sharpness": "{:.2f}".format(item.sharpness),
                    "eye_blink": "{:.4f}".format(item.eye_blink),
                    "eye_asymmetry": "{:.4f}".format(item.eye_asymmetry),
                    "mouth_abnormal": "{:.4f}".format(item.mouth_abnormal),
                    "expression_deviation": "{:.4f}".format(item.expression_deviation),
                    "eyes_good": item.eyes_good,
                    "mouth_good": item.mouth_good,
                    "quality_good": item.quality_good,
                    "overall_good": item.overall_good,
                    "x": int(item.bbox[0]), "y": int(item.bbox[1]),
                    "width": int(item.bbox[2]), "height": int(item.bbox[3]),
                })


def write_audit_json(
    path: Path,
    images: Sequence[ImageAnalysis],
    tracks: Sequence[PersonTrack],
    base: ImageAnalysis,
    records: Sequence[ReplacementRecord],
) -> None:
    data = {
        "base_image": base.path.name,
        "track_count": len(tracks),
        "images": [
            {
                "name": image.path.name,
                "detected_faces": len(image.observations),
                "good_faces": image_score(image)[0],
                "open_eye_faces": image_score(image)[1],
                "analyzed_faces": image_score(image)[2],
                "alignment_inliers": image.alignment_inliers,
                "alignment_ok": image.alignment_ok,
            }
            for image in images
        ],
        "replacements": [record.__dict__ for record in records],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_html_report(
    path: Path,
    images: Sequence[ImageAnalysis],
    tracks: Sequence[PersonTrack],
    base: ImageAnalysis,
    records: Sequence[ReplacementRecord],
    has_contact_sheet: bool,
) -> None:
    rows = []
    for image in sorted(images, key=image_score, reverse=True):
        score = image_score(image)
        selected = " class=\"selected\"" if image.index == base.index else ""
        rows.append(
            "<tr{}><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                selected, html.escape(image.path.name), len(image.observations), score[2], score[1], score[0],
                image.alignment_inliers,
            )
        )
    replacement_rows = []
    for record in records:
        replacement_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3f}</td>"
            "<td>{:.3f}</td><td>{}</td></tr>".format(
                html.escape(record.track_id), html.escape(record.region), html.escape(record.status),
                html.escape(record.donor_image or "-"), record.identity_similarity,
                record.outside_similarity, html.escape(record.reason),
            )
        )
    contact_sheet = (
        '<h2>成功替换放大对照</h2><p class="note">每组依次为母片、供体、最终结果。</p>'
        '<img src="replacement_review.jpg" alt="替换前后放大对照">'
        if has_contact_sheet else ""
    )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>全家福优化审核报告</title><style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;color:#202124;background:#f7f8fa}}
main{{max-width:1180px;margin:auto}} h1{{font-size:24px}} h2{{margin-top:28px;font-size:18px}}
.summary{{display:flex;gap:24px;flex-wrap:wrap;background:#fff;border:1px solid #ddd;padding:14px}}
.images{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} img{{width:100%;height:auto;background:#ddd}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px}} th,td{{padding:8px;border:1px solid #ddd;text-align:left}}
th{{background:#eef1f4}} tr.selected{{background:#e5f4e8}} .note{{color:#5f6368;font-size:13px}}
@media(max-width:720px){{.images{{grid-template-columns:1fr}} body{{margin:10px}} table{{font-size:11px}}}}
</style></head><body><main>
<h1>全家福优化审核报告</h1>
<div class="summary"><span>母片：<b>{base}</b></span><span>母片检测人物：<b>{base_faces}</b></span>
<span>跨帧轨迹：<b>{tracks}</b></span>
<span>成功替换：<b>{replaced}</b></span><span>保守跳过：<b>{skipped}</b></span></div>
<p class="note">标注图中绿色表示母片表情和质量通过，红色表示需要处理；E 为眼睛，M 为嘴部。最终成片仍建议按 100% 放大检查所有红色编号。</p>
<div class="images"><figure><img src="base_annotated.jpg"><figcaption>母片自动标注</figcaption></figure>
<figure><img src="final.jpg"><figcaption>最终输出</figcaption></figure></div>
{contact_sheet}
<h2>照片评分</h2><table><thead><tr><th>文件</th><th>检测</th><th>关键点</th><th>睁眼</th><th>整体通过</th><th>配准内点</th></tr></thead>
<tbody>{image_rows}</tbody></table>
<h2>替换审计</h2><table><thead><tr><th>人物</th><th>区域</th><th>状态</th><th>供体照片</th><th>身份相似度</th><th>其余区域相似度</th><th>说明</th></tr></thead>
<tbody>{replacement_rows}</tbody></table>
</main></body></html>""".format(
        base=html.escape(base.path.name), base_faces=len(base.observations), tracks=len(tracks),
        replaced=sum(record.status == "replaced" for record in records),
        skipped=sum(record.status != "replaced" for record in records),
        contact_sheet=contact_sheet,
        image_rows="".join(rows), replacement_rows="".join(replacement_rows),
    )
    path.write_text(document, encoding="utf-8")
