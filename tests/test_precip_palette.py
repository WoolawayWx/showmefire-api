import numpy as np
import pytest

from core import precip_palette as pp


def test_the_table_loads_with_one_band_per_step_and_dry_cells_white():
    qpf = pp.load_qpf_palette()
    assert qpf.levels[0] == 0.0 and qpf.levels[1] == 0.025 and qpf.levels[-1] == 40.0
    assert len(qpf.colors) == len(qpf.levels) - 1                 # a color per band, as BoundaryNorm needs
    assert qpf.colors[0] == "#FFFFFF"                             # below 0.025 in: no rain
    assert qpf.colors[1] == "#B6B6B6"                             # 0.025 in: 182 182 182
    assert qpf.over == "#B0340D"                                  # 40 in: 176 52 13, used for everything at or above it
    assert list(qpf.levels) == sorted(qpf.levels) and len(set(qpf.levels)) == len(qpf.levels)


def test_each_value_gets_the_color_of_the_step_it_has_reached():
    qpf = pp.load_qpf_palette()
    cmap, norm = qpf.cmap(), qpf.norm()

    def color(inches):
        return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in cmap(norm(inches))[:3]))

    assert color(0.0) == "#FFFFFF" and color(0.02) == "#FFFFFF"
    assert color(0.025) == "#B6B6B6" and color(0.09) == "#8B8183"            # 0.075 in: 139 129 131
    assert color(0.1) == "#AFF2C2" and color(0.12) == "#AFF2C2"              # 0.1 in: 175 242 194
    assert color(1.0) == "#FECE00" and color(1.2) == "#FECE00"               # 1 in: 254 206 0
    assert color(2.0) == "#F20500"                                            # 2 in: 242 5 0
    assert color(100.0) == qpf.over                                           # past the top step


def test_ticks_and_lines_are_real_levels_and_labels_are_two_lines():
    qpf = pp.load_qpf_palette()
    assert set(qpf.ticks) <= set(qpf.levels) and set(qpf.line_levels) <= set(qpf.levels)
    assert len(qpf.tick_labels) == len(qpf.ticks)
    assert qpf.tick_labels[0] == '0.00"\n0.0 mm' and qpf.tick_labels[3] == '0.50"\n12.7 mm'
    assert qpf.tick_labels[4] == '1.0"\n25 mm' and all("\n" in label for label in qpf.tick_labels)


def test_the_palette_file_is_validated(tmp_path, monkeypatch):
    for text, message in (("0.1 1 2\n", "Invalid"), ("0.1 1 2 300\n0.2 1 2 3\n", "out of range"),
                          ("0.2 1 2 3\n0.1 1 2 3\n", "increasing"), ("0.1 1 2 3\n", "at least two")):
        bad = tmp_path / "bad.pal"
        bad.write_text(text)
        monkeypatch.setattr(pp, "PALETTE_PATH", bad)
        pp.load_qpf_palette.cache_clear()
        with pytest.raises(ValueError, match=message):
            pp.load_qpf_palette()
    pp.load_qpf_palette.cache_clear()                              # leave the real table loaded for other tests


def test_the_forecast_v1_precipitation_style_uses_the_same_table():
    pp.load_qpf_palette.cache_clear()
    from forecast.map_palettes import style_for
    qpf = pp.load_qpf_palette()
    style = style_for("precipitation_increment")
    cmap, norm = style.build()
    assert style.levels == qpf.levels and style.colors == qpf.colors and style.extend == "max"
    assert tuple(cmap.get_over()) == tuple(cmap(norm(100.0))) and np.allclose(style.convert(np.array([25.4])), [1.0])
