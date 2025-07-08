import nncf
import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
import os

# --- Define paths (These names stay the same for compatibility with app.py) ---
fp16_model_dir = "tinyllama_fp16"
int8_model_dir = "tinyllama_int8" # We still call it int8, even though it's technically int4

print(f"Loading FP16 model from: {fp16_model_dir}")

# Load the FP16 model that was created by optimum-cli
try:
    fp16_model = OVModelForCausalLM.from_pretrained(fp16_model_dir)
except Exception as e:
    print(f"--- ERROR: Could not load the FP16 model. ---")
    print(f"Please make sure the '{fp16_model_dir}' folder exists and was created correctly.")
    print(f"DETAILS: {e}")
    input("\nPress Enter to exit.")
    exit()

# --- MODIFIED: Applying more aggressive INT4 quantization ---
print("Applying INT4 weight compression using NNCF... (This will be much smaller)")

# Use the nncf.compress_weights() function with the INT4 mode specified.
int4_model = nncf.compress_weights(
    fp16_model.model,
    mode=nncf.CompressWeightsMode.INT4_SYM  # This is the key change for aggressive compression
)

print(f"Quantization complete. Saving INT4 model to: {int8_model_dir}")

# Ensure the output directory exists
os.makedirs(int8_model_dir, exist_ok=True)

# Save the newly quantized model using the core ov.save_model function
ov.save_model(int4_model, os.path.join(int8_model_dir, "openvino_model.xml"))

# We must also save the configuration and tokenizer files so the app can load them
fp16_model.config.save_pretrained(int8_model_dir)
tokenizer = AutoTokenizer.from_pretrained(fp16_model_dir)
tokenizer.save_pretrained(int8_model_dir)

print(f"\nSuccess! Your highly optimized INT4 model is ready in the '{int8_model_dir}' folder.")