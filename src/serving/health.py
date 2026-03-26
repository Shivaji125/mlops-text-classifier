"""Health check endpoints for the API."""
import time

START_TIME = time.time()

def get_uptime():
    """Returns server uptime in seconds."""
    return round(time.time() - START_TIME, 2)

def check_model_health(model, tokenizer):
    """Runs a quick inference to verify model is working."""
    try:
        if model is None or tokenizer is None:
            return {"healthy": False, "reason": "Model or tokenizer not loaded"}
        
        import torch
        device = next(model.parameters()).device
        test_input =tokenizer(
            "test", return_tensors="pt", truncation=True,
            padding="max_length", max_length=128
        )
        test_input = {k: v.to(device) for k, v in test_input.items()}

        with torch.no_grad():
            output = model(**test_input)
        
        return {"healthy": True, "reason": "Model inference OK"}
    except Exception as e:
        return {"healthy": False, "reason": str(e)}