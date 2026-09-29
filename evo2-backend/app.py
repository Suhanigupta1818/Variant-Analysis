import spaces
import gradio as gr
import requests
import torch

from evo2 import Evo2

# ---------------------------------------------------------
# Evo2 7B model
# ---------------------------------------------------------
# Evo2 7B can run in bfloat16 without Transformer Engine / FP8.
# The model is loaded once when the Space starts.
print("Loading Evo2 7B...")
model = Evo2("evo2_7b")
print("Evo2 7B loaded successfully.")

WINDOW_SIZE = 8192

# These values come from the original project's BRCA1 calibration.
THRESHOLD = -0.0009178519
LOF_STD = 0.0015140239
FUNC_STD = 0.0009016589


def get_genome_sequence(
    position: int,
    genome: str,
    chromosome: str,
    window_size: int = WINDOW_SIZE,
):
    """Fetch an 8192 bp sequence window from the UCSC Genome API."""

    half_window = window_size // 2

    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1

    api_url = (
        "https://api.genome.ucsc.edu/getData/sequence"
        f"?genome={genome};chrom={chromosome};start={start};end={end}"
    )

    response = requests.get(api_url, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch genome sequence from UCSC API: "
            f"{response.status_code}"
        )

    genome_data = response.json()

    if "dna" not in genome_data:
        raise RuntimeError(
            f"UCSC API error: {genome_data.get('error', 'Unknown error')}"
        )

    sequence = genome_data.get("dna", "").upper()
    expected_length = end - start

    if len(sequence) != expected_length:
        raise RuntimeError(
            f"Unexpected sequence length: received {len(sequence)}, "
            f"expected {expected_length}"
        )

    return sequence, start


def analyze_variant(
    relative_pos_in_window: int,
    reference: str,
    alternative: str,
    window_seq: str,
):
    """Score reference and alternate sequences with Evo2."""

    if len(alternative) != 1 or alternative not in {"A", "C", "G", "T"}:
        raise ValueError("Alternative must be one DNA base: A, C, G, or T.")

    if reference not in {"A", "C", "G", "T"}:
        raise ValueError(
            f"Reference base returned by UCSC is '{reference}', "
            "which is not a standard A/C/G/T base."
        )

    if relative_pos_in_window < 0 or relative_pos_in_window >= len(window_seq):
        raise ValueError("Variant position is outside the fetched sequence window.")

    var_seq = (
        window_seq[:relative_pos_in_window]
        + alternative
        + window_seq[relative_pos_in_window + 1:]
    )

    # Evo2 returns sequence likelihood scores.
    ref_score = model.score_sequences([window_seq])[0]
    var_score = model.score_sequences([var_seq])[0]

    delta_score = float(var_score - ref_score)

    if delta_score < THRESHOLD:
        prediction = "Likely pathogenic"
        confidence = min(
            1.0,
            abs(delta_score - THRESHOLD) / LOF_STD
        )
    else:
        prediction = "Likely benign"
        confidence = min(
            1.0,
            abs(delta_score - THRESHOLD) / FUNC_STD
        )

    return {
        "reference": reference,
        "alternative": alternative,
        "delta_score": delta_score,
        "prediction": prediction,
        "classification_confidence": float(confidence),
    }


# ---------------------------------------------------------
# GPU inference
# ---------------------------------------------------------
# ZeroGPU dynamically allocates a GPU while this function runs.
@spaces.GPU(duration=180)
def analyze_single_variant(
    variant_position: int,
    alternative: str,
    genome: str,
    chromosome: str,
):
    """Analyze one SNV using Evo2 7B."""

    try:
        variant_position = int(variant_position)
    except (TypeError, ValueError):
        raise gr.Error("Variant position must be an integer.")

    alternative = str(alternative).strip().upper()
    genome = str(genome).strip()
    chromosome = str(chromosome).strip()

    if variant_position <= 0:
        raise gr.Error("Variant position must be greater than 0.")

    if not genome:
        raise gr.Error("Please enter a genome assembly, e.g. hg38.")

    if not chromosome:
        raise gr.Error("Please enter a chromosome, e.g. chr17.")

    if alternative not in {"A", "C", "G", "T"}:
        raise gr.Error("Alternative must be A, C, G, or T.")

    try:
        window_seq, seq_start = get_genome_sequence(
            position=variant_position,
            genome=genome,
            chromosome=chromosome,
            window_size=WINDOW_SIZE,
        )

        relative_pos = variant_position - 1 - seq_start

        if relative_pos < 0 or relative_pos >= len(window_seq):
            raise ValueError(
                f"Variant position is outside the fetched window "
                f"(start={seq_start + 1}, "
                f"end={seq_start + len(window_seq)})."
            )

        reference = window_seq[relative_pos]

        result = analyze_variant(
            relative_pos_in_window=relative_pos,
            reference=reference,
            alternative=alternative,
            window_seq=window_seq,
        )

        result["position"] = variant_position
        result["genome"] = genome
        result["chromosome"] = chromosome
        result["window_start"] = seq_start + 1
        result["window_end"] = seq_start + len(window_seq)

        return result

    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Analysis failed: {exc}")


# ---------------------------------------------------------
# Gradio UI / API
# ---------------------------------------------------------
with gr.Blocks(title="Variant Analysis - Evo2 7B") as demo:
    gr.Markdown(
        """
        # 🧬 Variant Analysis with Evo2 7B

        Enter a genomic position and alternate allele to estimate
        variant effect using Evo2 7B sequence likelihoods.

        **Genome:** e.g. `hg38`  
        **Chromosome:** e.g. `chr17`  
        **Alternative:** `A`, `C`, `G`, or `T`
        """
    )

    with gr.Row():
        variant_position = gr.Number(
            label="Variant Position",
            value=43119628,
            precision=0,
        )
        alternative = gr.Textbox(
            label="Alternative",
            value="G",
            max_lines=1,
        )

    with gr.Row():
        genome = gr.Textbox(
            label="Genome Assembly",
            value="hg38",
            max_lines=1,
        )
        chromosome = gr.Textbox(
            label="Chromosome",
            value="chr17",
            max_lines=1,
        )

    analyze_button = gr.Button("Analyze Variant", variant="primary")

    output = gr.JSON(label="Prediction Result")

    analyze_button.click(
        fn=analyze_single_variant,
        inputs=[
            variant_position,
            alternative,
            genome,
            chromosome,
        ],
        outputs=output,
        api_name="analyze_variant",
    )

    gr.Markdown(
        """
        **Note:** This is an academic/demo implementation of Evo2-based
        zero-shot variant scoring. The pathogenic/benign labels use the
        calibration thresholds from the original project and should not
        be interpreted as a clinical diagnosis.
        """
    )


if __name__ == "__main__":
    demo.launch()
