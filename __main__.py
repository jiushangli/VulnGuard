"""
VulnGuard CLI Entry Point.

Usage:
    vulnguard audit --config vulnguard.yaml --target /path/to/repo
    vulnguard analyze --config vulnguard.yaml --target /path/to/repo --phase intelligence
    vulnguard report --db /path/to/vulnguard.db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from config.settings import VulnGuardConfig, default_config


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the CLI."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_config(config_path: str | None, target_path: str | None, language: str | None) -> VulnGuardConfig:
    """
    Load configuration from a YAML file if provided, otherwise use defaults.
    Command-line arguments override config file values.
    """
    if config_path:
        config = VulnGuardConfig.from_yaml(config_path)
    else:
        config = default_config()

    # Override with CLI arguments
    if target_path:
        config.target.repo_path = target_path
    if language:
        config.target.language = language

    return config


async def run_audit(config: VulnGuardConfig, phase: str | None = None) -> int:
    """Run a full or partial audit using the Orchestrator."""
    from orchestrator import Orchestrator

    orchestrator = Orchestrator(config=config)
    report = await orchestrator.run(phase=phase)

    # Output report
    print("\n" + "=" * 60)
    print(report.executive_summary)
    print()

    # Save detailed report to file
    report_dir = Path(config.security.poc_directory).parent / "vulnguard-reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = report.started_at.replace(":", "-").replace(".", "_") if report.started_at else "unknown"
    report_file = report_dir / f"audit-report-{timestamp}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.detailed_report)

    # Also save as JSON for programmatic consumption
    json_file = report_dir / f"audit-report-{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "target": report.target,
            "language": report.language,
            "total_findings": report.total_findings,
            "vulnerabilities": [
                {
                    "fact_id": v.fact_id,
                    "vuln_type": v.vuln_type,
                    "description": v.description,
                    "confidence": v.confidence,
                    "severity": v.severity,
                    "remediation": v.remediation,
                }
                for v in report.vulnerabilities
            ],
            "ruled_outs": [
                {
                    "fact_id": r.fact_id,
                    "vuln_type": r.vuln_type,
                    "ruled_out": r.ruled_out,
                    "remaining_risk": r.remaining_risk,
                    "confidence": r.confidence,
                }
                for r in report.ruled_outs
            ],
            "statistics": report.statistics,
            "started_at": report.started_at,
            "completed_at": report.completed_at,
        }, f, indent=2, default=str)

    print(f"\nDetailed report saved to: {report_file}")
    print(f"JSON report saved to: {json_file}")

    # Return exit code based on findings
    critical_count = sum(1 for v in report.vulnerabilities if v.severity == "critical")
    high_count = sum(1 for v in report.vulnerabilities if v.severity == "high")

    if critical_count > 0:
        print(f"\n⚠  CRITICAL: {critical_count} critical vulnerabilities found!")
        return 2
    elif high_count > 0:
        print(f"\n⚠  WARNING: {high_count} high-severity vulnerabilities found!")
        return 1
    else:
        print("\n✓  No critical or high-severity vulnerabilities found.")
        return 0


def run_report(db_path: str, target: str = "", language: str = "") -> int:
    """Generate a report from an existing VulnKB database."""
    from orchestrator import generate_report_from_db

    report = generate_report_from_db(
        db_path=db_path,
        target=target,
        language=language,
    )

    print("\n" + report.executive_summary)
    print("\n" + report.detailed_report)

    # Return exit code based on findings
    critical_count = sum(1 for v in report.vulnerabilities if v.severity == "critical")
    high_count = sum(1 for v in report.vulnerabilities if v.severity == "high")

    if critical_count > 0:
        return 2
    elif high_count > 0:
        return 1
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vulnguard",
        description="VulnGuard — Business Logic Vulnerability Mining & Verification Agent Framework",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── audit command ───
    audit_parser = subparsers.add_parser(
        "audit",
        help="Run a full vulnerability audit on a target repository",
    )
    audit_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    audit_parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to the target repository to audit",
    )
    audit_parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Primary language of the target repository (default: python)",
    )
    audit_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database for VulnKB persistence (default: in-memory)",
    )

    # ─── analyze command ───
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run a specific audit phase (intelligence, mining, or verification)",
    )
    analyze_parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    analyze_parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        help="Path to the target repository to analyze",
    )
    analyze_parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Primary language of the target repository",
    )
    analyze_parser.add_argument(
        "--phase", "-p",
        type=str,
        choices=["intelligence", "mining", "verification"],
        default="intelligence",
        help="Audit phase to run (default: intelligence)",
    )
    analyze_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database for VulnKB persistence",
    )

    # ─── report command ───
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a report from an existing VulnKB database",
    )
    report_parser.add_argument(
        "--db", "-d",
        type=str,
        required=True,
        help="Path to the VulnKB SQLite database file",
    )
    report_parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Target repository path (for report header)",
    )
    report_parser.add_argument(
        "--language",
        type=str,
        default="",
        help="Language (for report header)",
    )

    # ─── version command ───
    subparsers.add_parser(
        "version",
        help="Show VulnGuard version",
    )

    args = parser.parse_args()

    # Handle no command
    if args.command is None:
        parser.print_help()
        return 1

    # Version command
    if args.command == "version":
        from vulnguard import __version__
        print(f"VulnGuard v{__version__}")
        return 0

    # Setup logging
    setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    # ─── audit command ───
    if args.command == "audit":
        config = load_config(args.config, args.target, args.language)

        # Override db_path if specified
        if args.db_path:
            config.vulnkb.db_path = args.db_path
            # For file-based DB, also persist
            config.vulnkb.storage = "sqlite"

        return asyncio.run(run_audit(config))

    # ─── analyze command ───
    if args.command == "analyze":
        config = load_config(args.config, args.target, args.language)

        if args.db_path:
            config.vulnkb.db_path = args.db_path
            config.vulnkb.storage = "sqlite"

        return asyncio.run(run_audit(config, phase=args.phase))

    # ─── report command ───
    if args.command == "report":
        return run_report(args.db, args.target, args.language)

    return 0


if __name__ == "__main__":
    sys.exit(main())