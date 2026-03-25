from hex_ai.utils.ladder_templates.prefill_geometry import (
    DetectedImageCell,
    build_prefill_annotation_payload,
    infer_offset_grid,
)


def test_infer_offset_grid_recovers_local_offset_coordinates():
    radius = 20.0
    horizontal_pitch = radius * (3 ** 0.5)
    vertical_pitch = radius * 1.5
    origin_x = 120.0
    origin_y = 90.0
    cells = [
        DetectedImageCell(center_x=origin_x + horizontal_pitch * (1 + 0 / 2), center_y=origin_y + vertical_pitch * 0, state="red"),
        DetectedImageCell(center_x=origin_x + horizontal_pitch * (2 + 0 / 2), center_y=origin_y + vertical_pitch * 0, state="red"),
        DetectedImageCell(center_x=origin_x + horizontal_pitch * (0 + 1 / 2), center_y=origin_y + vertical_pitch * 1, state="plus"),
        DetectedImageCell(center_x=origin_x + horizontal_pitch * (1 + 1 / 2), center_y=origin_y + vertical_pitch * 1, state="empty"),
        DetectedImageCell(center_x=origin_x + horizontal_pitch * (2 + 1 / 2), center_y=origin_y + vertical_pitch * 1, state="blue"),
    ]

    geometry = infer_offset_grid(cells, radius_hint=radius)

    assert geometry.rows == 2
    assert geometry.cols == 3
    assert geometry.radius == radius
    assert tuple((cell.row, cell.col, cell.state) for cell in geometry.cells) == (
        (0, 1, "red"),
        (0, 2, "red"),
        (1, 0, "plus"),
        (1, 1, "empty"),
        (1, 2, "blue"),
    )


def test_build_prefill_annotation_payload_uses_inferred_geometry():
    geometry = infer_offset_grid(
        [
            DetectedImageCell(center_x=100.0, center_y=100.0, state="red"),
            DetectedImageCell(center_x=134.641016151, center_y=100.0, state="blue"),
        ],
        radius_hint=20.0,
    )

    payload = build_prefill_annotation_payload(
        geometry,
        image_name="sample.png",
        image_width=640,
        image_height=480,
        family="unit_test",
        template_name="prefill_sample",
        open_left=False,
        open_right=False,
    )

    assert payload["metadata"]["name"] == "prefill_sample"
    assert payload["metadata"]["family"] == "unit_test"
    assert payload["image_overlay"]["image_name"] == "sample.png"
    assert payload["image_overlay"]["image_offset_x"] == 0.0
    assert payload["image_overlay"]["image_offset_y"] == 0.0
    assert payload["grid"]["rows"] == geometry.rows
    assert payload["grid"]["cols"] == geometry.cols
    assert payload["cells"] == [
        {"row": 0, "col": 0, "state": "red"},
        {"row": 0, "col": 1, "state": "blue"},
    ]
