#!/usr/bin/env python3

from huggingface_hub import HfApi
import sys


def check_model_access(model_id="meta-llama/Meta-Llama-3-8B"):
    """
    Check if you have access to a Hugging Face model
    """
    print(f"Checking access to model: {model_id}")
    print("-" * 50)

    try:
        # Initialize HF API
        api = HfApi()

        # Try to get model info
        model_info = api.model_info(model_id)

        print("✅ SUCCESS: You have access to the model!")
        print(f"Model ID: {model_info.modelId}")
        print(f"Downloads: {model_info.downloads}")
        print(f"Likes: {model_info.likes}")
        print(f"Library: {model_info.library_name}")

        if model_info.gated:
            print("⚠️  Note: This model is gated - you may need special permissions")
        # model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        return True

    except Exception as e:
        error_msg = str(e).lower()

        if "not found" in error_msg or "404" in error_msg:
            print("❌ ERROR: Model not found")
        elif "access" in error_msg or "unauthorized" in error_msg or "403" in error_msg:
            print(
                "❌ ERROR: Access denied - you don't have permission to access this model"
            )
            print("💡 To get access:")
            print("   1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B")
            print("   2. Request access if it's gated")
            print("   3. Make sure you're logged in with: huggingface-cli login")
            raise e
        else:
            print(f"❌ ERROR: {e}")

        return False


def check_authentication():
    """Check if user is authenticated with Hugging Face"""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not authenticated with Hugging Face")
        print("💡 Run: huggingface-cli login")
        return False


if __name__ == "__main__":
    print("Hugging Face Model Access Checker")
    print("=" * 50)

    # Check authentication first
    if not check_authentication():
        sys.exit(1)

    print()

    # Check model access
    model_id = "meta-llama/Meta-Llama-3-8B"
    if len(sys.argv) > 1:
        model_id = sys.argv[1]

    success = check_model_access(model_id)

    if success:
        print("\n🎉 You can now use this model in your code!")
        print("Example usage:")
        print("from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{model_id}')")
        print(f"model = AutoModelForCausalLM.from_pretrained('{model_id}')")
    else:
        print("\n❌ Model access check failed")
        sys.exit(1)
