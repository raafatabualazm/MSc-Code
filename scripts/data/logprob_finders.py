import requests
import json
import time

# --- CONFIGURATION ---
API_KEY = "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"  # Replace with your OpenRouter API key
MODEL_ID = "moonshotai/kimi-k2-thinking"  # Replace with the model you want to test
# ---------------------

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://localhost.com", # Required by OpenRouter
    "X-Title": "Logprobs Tester"
}

def get_all_providers():
    """Fetches the list of all available providers from OpenRouter."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/providers")
        response.raise_for_status()
        data = response.json()
        return [p['slug'] for p in data['data']]
    except Exception as e:
        print(f"Error fetching providers: {e}")
        return []

def test_provider(provider_slug, model_id):
    """
    Tests a specific provider for logprobs support by forcing routing
    and disabling fallbacks.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 4096,
        "logprobs": True,        # Request logprobs
        "top_logprobs": 1,       # Required param when logprobs is True
        "reasoning": {"enabled": True},  # Request reasoning tokens
        "provider": {
            "order": [provider_slug],
            "allow_fallbacks": False # CRITICAL: Forces this specific provider only
        }
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=HEADERS,
            json=payload,
        )
        
        # Parse response
        if response.status_code == 200:
            result = response.json()
            choices = result.get('choices', [])
            if not choices:
                return "No choices returned"
                
            # Check for logprobs in the response
            logprobs = choices[0].get('logprobs')
            
            if logprobs:
                return "SUPPORTED"
            else:
                return "Model served, but logprobs NOT returned"
        
        else:
            error_msg = response.json().get('error', {}).get('message', '')
            # Common errors for "provider doesn't support this model"
            if "No available providers" in error_msg or "endpoint" in error_msg.lower():
                return "Model not supported by provider"
            return f"Error: {response.status_code} - {error_msg}"

    except Exception as e:
        return f"Request Failed: {str(e)}"

def main():
    print(f"--- Testing Logprobs Support for: {MODEL_ID} ---\n")
    
    # 1. Get all providers
    print("Fetching provider list...")
    providers = get_all_providers()
    print(f"Found {len(providers)} total providers on OpenRouter. Testing each one...\n")
    
    supported_providers = []
    no_logprobs_providers = []

    # 2. Iterate and Test
    # Limit specifically to known reliable providers first if you want to save time, 
    # but this loop tests everyone.
    for i, slug in enumerate(providers):
        print(f"[{i+1}/{len(providers)}] Testing provider: {slug.ljust(20)} ... ", end="", flush=True)
        
        status = test_provider(slug, MODEL_ID)
        print(status)
        
        if status == "SUPPORTED":
            supported_providers.append(slug)
        elif status == "Model served, but logprobs NOT returned":
            no_logprobs_providers.append(slug)
            
        # Rate limit safety (optional)
        time.sleep(0.2)

    # 3. Summary
    print("\n" + "="*50)
    print(f"RESULTS FOR {MODEL_ID}")
    print("="*50)
    
    if supported_providers:
        print(f"\n✅ Providers SUPPORTING logprobs ({len(supported_providers)}):")
        for p in supported_providers:
            print(f" - {p}")
    else:
        print("\n❌ No providers found that support logprobs for this model.")

    if no_logprobs_providers:
        print(f"\n⚠️  Providers serving model but MISSING logprobs ({len(no_logprobs_providers)}):")
        for p in no_logprobs_providers:
            print(f" - {p}")

if __name__ == "__main__":
    main()