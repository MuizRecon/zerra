"""Regression tests for two security fixes:

1. Forged GitHub webhooks could exfiltrate a stored GitHub token by
   naming a clone_url whose host merely *contains* "github.com"
   (e.g. "github.com.attacker.example"), and an empty repository
   full_name would match every monitored repo via a substring check.
2. The SAST / secrets scanners followed symlinks, so a repo could
   contain a symlink pointing outside the clone directory and have
   that external file's contents read and returned as a finding.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os

import pytest
from fastapi.testclient import TestClient
from unittest import mock

from agent.api import app
from agent.db.database import get_db
from agent.scanner.models import ScanConfig
from agent.scanner.repo_scanner import RepoScanner
from agent.scanner.sast import scan_directory_with_rules
from agent.scanner.secrets import scan_directory_for_secrets


FORGED_PAYLOAD = {
    "ref": "refs/heads/main",
    "repository": {
        "full_name": "",
        "clone_url": "https://github.com.attacker.example/x.git",
    },
}


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


class TestWebhookSignatureEnforcement:
    def test_rejects_when_no_secret_configured(self, monkeypatch) -> None:
        monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
        client = TestClient(app)
        response = client.post("/v1/webhooks/github", json=FORGED_PAYLOAD)
        assert response.status_code == 401

    def test_rejects_missing_signature_header(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "testsecret")
        client = TestClient(app)
        response = client.post("/v1/webhooks/github", json=FORGED_PAYLOAD)
        assert response.status_code == 401

    def test_rejects_incorrect_signature(self, monkeypatch) -> None:
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "testsecret")
        client = TestClient(app)
        body = json.dumps(FORGED_PAYLOAD).encode()
        response = client.post(
            "/v1/webhooks/github",
            content=body,
            headers={
                "content-type": "application/json",
                "X-Hub-Signature-256": "sha256=" + "0" * 64,
            },
        )
        assert response.status_code == 401

    def test_valid_signature_with_empty_full_name_matches_nothing(self, monkeypatch) -> None:
        """Even with a VALID signature, an empty repo full_name must not
        match every monitored repo via the old substring check."""
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "testsecret")
        client = TestClient(app)
        body = json.dumps(FORGED_PAYLOAD).encode()
        response = client.post(
            "/v1/webhooks/github",
            content=body,
            headers={
                "content-type": "application/json",
                "X-Hub-Signature-256": _sign("testsecret", body),
            },
        )
        assert response.status_code == 200
        assert response.json() == {"processed": 0, "results": []}

    def test_matched_repo_is_scanned_by_its_stored_url_not_payload_url(self, monkeypatch) -> None:
        """A legitimate, correctly-signed webhook for a real monitored repo
        must scan the URL we have on file, never the URL from the payload."""
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "testsecret")
        db = get_db()
        db.save_repo({
            "url": "https://github.com/someorg/somerepo",
            "branch": "main",
            "scan_mode": "quick",
            "github_token": "ghp_DUMMYTOKEN00000000000000000000000000",
        })

        payload = {
            "ref": "refs/heads/main",
            "repository": {
                "full_name": "someorg/somerepo",
                "clone_url": "https://github.com.attacker.example/x.git",
            },
        }
        body = json.dumps(payload).encode()

        with mock.patch.object(RepoScanner, "scan_async", autospec=True) as fake_scan:
            async def _fake(self, config):
                assert config.repo_url == "https://github.com/someorg/somerepo"
                assert "attacker.example" not in config.repo_url
                from agent.scanner.models import ScanResult, ScanStatus
                return ScanResult(
                    repo_url=config.repo_url, branch="main",
                    mode=config.mode, status=ScanStatus.COMPLETED,
                    duration_seconds=0.0,
                )
            fake_scan.side_effect = _fake

            client = TestClient(app)
            response = client.post(
                "/v1/webhooks/github",
                content=body,
                headers={
                    "content-type": "application/json",
                    "X-Hub-Signature-256": _sign("testsecret", body),
                },
            )
        assert response.status_code == 200
        assert response.json()["processed"] == 1


class TestCloneTokenHostValidation:
    """agent.scanner.repo_scanner.RepoScanner._clone_repo must only inject
    the GitHub token when the URL's actual hostname is github.com."""

    def _captured_clone_cmd(self, repo_url: str, token: str) -> list[str]:
        scanner = RepoScanner()
        config = ScanConfig(repo_url=repo_url, github_token=token)
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            import subprocess
            return subprocess.CompletedProcess(cmd, 1, "", "stub failure")

        with mock.patch("subprocess.run", fake_run):
            try:
                scanner._clone_repo(config)
            except RuntimeError:
                pass
        return captured["cmd"]

    def test_token_not_injected_for_lookalike_host(self) -> None:
        cmd = self._captured_clone_cmd(
            "https://github.com.attacker.example/x.git",
            "ghp_SECRETTOKEN00000000000000000000000000",
        )
        joined = " ".join(cmd)
        assert "ghp_SECRETTOKEN" not in joined

    def test_token_injected_for_real_github_host(self) -> None:
        cmd = self._captured_clone_cmd(
            "https://github.com/someorg/somerepo",
            "ghp_SECRETTOKEN00000000000000000000000000",
        )
        joined = " ".join(cmd)
        assert "ghp_SECRETTOKEN" in joined


class TestScannersSkipSymlinks:
    """Secrets and SAST scanners must not follow symlinks out of the repo."""

    @pytest.fixture()
    def symlinked_repo(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        secret_file = outside / "server_secrets.env"
        secret_file.write_text(
            'AWS_SECRET_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n'
        )

        repo = tmp_path / "evil-repo"
        repo.mkdir()
        (repo / "innocent.txt").symlink_to(secret_file)
        (repo / "a.py").write_text("print(1)\n")
        return repo

    def test_secrets_scanner_ignores_symlinked_file(self, symlinked_repo) -> None:
        findings = scan_directory_for_secrets(symlinked_repo)
        assert findings == []

    def test_sast_scanner_ignores_symlinked_file(self, symlinked_repo) -> None:
        findings = scan_directory_with_rules(symlinked_repo)
        assert not any(f.file_path == "innocent.txt" for f in findings)
