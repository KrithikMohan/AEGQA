"""Build the PR #10 PDF from the checked compatibility record.

Document authoring dependency: reportlab (available in the Codex bundled runtime).
Research runtime requirements do not need this document-generation dependency.
Run the compatibility audit first: python -m aeqga.analysis.pr10_compatibility
    --output docs/PR10_COMPATIBILITY.json
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

ROOT = Path(__file__).resolve().parents[1]


def build():
    report = json.loads((ROOT / "docs/PR10_COMPATIBILITY.json").read_text())
    pilot, generation = report["pilot"], report["generation"]
    output = ROOT / "output/pdf/PR10_SUMMARY.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    navy, teal, muted = colors.HexColor("#14324B"), colors.HexColor("#007F82"), colors.HexColor("#506575")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=25,
                              leading=29, textColor=navy, spaceAfter=12))
    styles.add(ParagraphStyle("Deck", fontName="Helvetica", fontSize=11,
                              leading=16, textColor=muted, spaceAfter=12))
    styles.add(ParagraphStyle("Section", fontName="Helvetica-Bold", fontSize=13,
                              leading=17, textColor=teal, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle("ReportBody", fontName="Helvetica", fontSize=10,
                              leading=14, spaceAfter=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle("SmallReport", fontName="Helvetica", fontSize=8,
                              leading=11, textColor=muted, spaceAfter=5))
    story = []

    def paragraph(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths=(3.8*inch, 2.6*inch)):
        data = [[Paragraph(escape(str(v)), styles["ReportBody"]) for v in row] for row in rows]
        block = Table(data, colWidths=widths, hAlign="LEFT")
        block.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E5F0F1")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F8FA")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 9),
            ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        story.append(block)

    paragraph("PR #10 | Research summary", "SmallReport")
    paragraph("IBM Kingston experiments", "ReportTitle")
    paragraph("Archived hardware results, reproduced with the current AEQGA implementation.<br/>Review date: October 30, 2026", "Deck")
    paragraph('PR #10, <i>Add IBM Kingston experimental results and analysis</i>, merged on '
              'October 30, 2026 at 05:13 UTC (commit 599927f; content commit 7b1149a). '
              'Its nine files preserve two job ledgers, recovered pilot counts, a population '
              'archive, two plots and three analysis/recovery scripts. The recorded job '
              'submission timestamps are October 11, 2026. This is experimental evidence '
              'from a short pilot and one generation.')
    paragraph("1. Single-circuit pilot", "Section")
    paragraph(f'Job <b>{pilot["job_id"]}</b>: one four-qubit circuit, {pilot["shots"]} shots, '
              'seed 42, input values drawn with default_rng(42) from [0.1, 1.0]. '
              'Crossover and mutation probabilities were both 1.0.')
    table([
        ["Recomputed metric", "Result"],
        ["Observed basis states", f'{pilot["observed_states"]} / 16'],
        ["Total variation distance (TVD)", f'{pilot["tvd"]:.9f}'],
        ["Largest probability difference", f'{pilot["max_probability_error"]:.9f}'],
        ["Ideal finite-shot TVD: mean / 99th percentile",
         f'{pilot["finite_shot_mean_tvd"]:.6f} / {pilot["finite_shot_p99"]:.6f}'],
        ["Ideal trials with TVD at least as large", "0 / 10,000"],
    ])
    paragraph("The measured discrepancy exceeds the reconstructed ideal finite-shot baseline. "
              "This is consistent with additional device error, conditional on the reconstructed "
              "logical input. Zero exceedances is a finite Monte Carlo result, not an exact "
              "zero probability. TVD does not determine gate fidelity or identify individual error sources.")
    image = ROOT / "outputs/png/ibm_first_comparison.png"
    if image.exists():
        story.append(Image(str(image), width=6.4*inch, height=2.67*inch))
        paragraph("Original PR #10 figure, retained unchanged; numerical values independently recomputed.", "SmallReport")

    story.append(PageBreak())
    paragraph("Compatibility and interpretation", "ReportTitle")
    paragraph("2. Complete generation-1 experiment", "Section")
    paragraph(f'Job <b>{generation["job_id"]}</b>: population 32, one update, seed 42, '
              'crossover/mutation probabilities 0.5, four circuits of widths [4, 3, 4, 3], '
              '1024 shots each. The current calibrated 1701-row Pantheon+ likelihood uses '
              'MU_SH0ES, full STAT+SYS covariance, zHD/zHEL and a 300-node distance grid.')
    table([
        ["Current replay result", "Value"],
        ["Best H0 / Omega_m", f'{generation["best_x"][0]:.8f} / {generation["best_x"][1]:.8f}'],
        ["Best chi-squared", f'{generation["best_chi2"]:.9f}'],
        ["Gap above configured classical minimum", f'{generation["delta_chi2"]:.6f}'],
        ["Objective evaluations", generation["evaluations"]],
        ["Archived initial/final populations: replay error", "0 in all four arrays"],
        ["Preserved classical elites", "All eight retained exactly"],
    ])
    paragraph("Hardware replay and the ideal finite-shot run retain the same best individual. "
              "That best individual comes from classical elitism; it does not establish QPU "
              "improvement or convergence. One generation does not reproduce the paper's "
              "300-run, 50-generation statistical experiment.")
    paragraph("3. Compatibility work and conflict resolution", "Section")
    paragraph("The old population replay helper called execute_generation(), while the current "
              "executor contract uses run(circuits, generation, optimizer_state). The updated "
              "historical replay uses that contract and validates widths, count totals and "
              "generation number. Recovery now accepts completed schema-v1 count ledgers "
              "offline. The pilot analysis uses the actual archived input convention rather "
              "than the older near-uniform H0 example.")
    paragraph("The approved PR #9 resolution preserves the seven current implementations "
              "and removes seven obsolete overlay/install/helper files. All nine PR #10 "
              "artifacts were already present on main. Historical records and exported "
              "research plots are retained; notebook execution outputs are cleared.")
    paragraph("4. Verification boundary", "Section")
    paragraph("Regression tests check the pilot statistics, all four archived population arrays, "
              "offline legacy recovery, corruption rejection and refusal to resume legacy "
              "records as current hardware jobs. Modern tests cover submitted-job recovery "
              "and restart boundaries. No physical QPU jobs were submitted for this review.")
    paragraph("The legacy compiled-circuit hashes cannot be verified because the original ISA/QPY "
              "archives and optimizer provenance were not supplied. Exact population replay "
              "shows software compatibility under the documented seed/settings; it does not "
              "authenticate the historical device circuit. Start new hardware experiments in "
              "a new directory using the maintained scripts/run_ibm_generation.py workflow.")
    paragraph("Sources and reproducibility", "Section")
    paragraph('<link href="https://github.com/KrithikMohan/AEQGA/pull/10" color="#007F82">'
              'GitHub PR #10</link> | '
              '<link href="https://github.com/KrithikMohan/AEQGA/pull/9" color="#007F82">GitHub PR #9</link><br/>'
              'Numeric evidence, package versions and source-artifact SHA-256 values: '
              'docs/PR10_COMPATIBILITY.json. Recompute with:<br/>'
              'python -m aeqga.analysis.pr10_compatibility --output docs/PR10_COMPATIBILITY.json', "SmallReport")

    def footer(canvas, doc):
        canvas.setStrokeColor(teal)
        canvas.line(45, 38, 567, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(45, 25, "AEQGA | PR #10 audit | October 30, 2026")
        canvas.drawRightString(567, 25, str(doc.page))

    doc = SimpleDocTemplate(str(output), pagesize=(612, 792),
                            leftMargin=45, rightMargin=45, topMargin=40, bottomMargin=48,
                            title="PR #10 - IBM Kingston experiments and compatibility",
                            author="AEQGA repository")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    build()
