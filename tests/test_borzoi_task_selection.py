from pathlib import Path

import yaml

from pocketreg.borzoi.task_selection import select_targets


def test_select_targets_metadata_bootstrap(tmp_path: Path):
    targets = tmp_path / "targets.txt"
    targets.write_text(
        "\tidentifier\tdescription\n"
        "0\tENC0+\tRNA:K562\n"
        "1\tENC0-\tRNA:K562\n"
        "2\tENC1+\tRNA:K562\n"
        "3\tOTHER\tATAC:K562\n"
    )
    config = yaml.safe_load(
        """
primary_targets:
  indices: [1]
  max_primary: 1
aux_targets:
  k_aux: 2
  exclude_primary: true
"""
    )
    selection = select_targets(targets, config)
    assert selection["primary_targets"][0]["index"] == 1
    assert [row["index"] for row in selection["aux_targets"]] == [0, 2]
    assert selection["selection_mode"] == "metadata_only_pending_rich_cache"
