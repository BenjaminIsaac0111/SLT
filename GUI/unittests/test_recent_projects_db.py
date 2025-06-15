import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from GUI.models import RecentProjectsDB as rdb


def test_save_and_list_recent_paths(tmp_path, monkeypatch):
    db_file = tmp_path / "recent.db"
    monkeypatch.setattr(rdb, "_DB_PATH", db_file)

    rdb.save_path("one.slt")
    rdb.save_path("two.slt")

    paths = rdb.get_recent_paths(2)
    assert paths == ["two.slt", "one.slt"]

