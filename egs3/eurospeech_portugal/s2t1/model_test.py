#!/usr/bin/env python3
"""Test script for OWSMFinetune model."""

import torch
from model import OWSMFinetune


def test_owsm_finetune():
    """Test the OWSMFinetune model with dummy data."""
    
    # Model configuration
    model_tag = "espnet/owsm_v3.1_ebf_base"  # Replace with your desired model tag
    
    print("Loading OWSMFinetune model...")
    model = OWSMFinetune(model_tag)
    print(f"Model loaded successfully: {type(model.model)}")
    
    # Create dummy input data
    batch_size = 2
    max_speech_length = 16000 * 10  # 10 seconds at 16kHz
    max_text_length = 50
    
    # Speech inputs
    speech = torch.randn(batch_size, max_speech_length)
    speech_lengths = torch.tensor([max_speech_length, max_speech_length // 2])
    
    # Text inputs (token IDs)
    text = torch.randint(0, 1000, (batch_size, max_text_length))
    text_lengths = torch.tensor([max_text_length, max_text_length // 2])
    
    # CTC text inputs
    text_ctc = torch.randint(0, 1000, (batch_size, max_text_length))
    text_ctc_lengths = torch.tensor([max_text_length, max_text_length // 2])
    
    # Previous text inputs (for teacher forcing)
    text_prev = torch.randint(0, 1000, (batch_size, max_text_length))
    text_prev_lengths = torch.tensor([max_text_length, max_text_length // 2])
    
    print("\nInput shapes:")
    print(f"  speech: {speech.shape}")
    print(f"  speech_lengths: {speech_lengths}")
    print(f"  text: {text.shape}")
    print(f"  text_lengths: {text_lengths}")
    print(f"  text_ctc: {text_ctc.shape}")
    print(f"  text_ctc_lengths: {text_ctc_lengths}")
    print(f"  text_prev: {text_prev.shape}")
    print(f"  text_prev_lengths: {text_prev_lengths}")
    
    # Test forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():  # Disable gradient computation for testing
            output = model.forward(
                speech=speech,
                speech_lengths=speech_lengths,
                text=text,
                text_lengths=text_lengths,
                text_ctc=text_ctc,
                text_ctc_lengths=text_ctc_lengths,
                text_prev=text_prev,
                text_prev_lengths=text_prev_lengths,
            )
        
        print("Forward pass successful!")
        print(f"\nOutput type: {type(output)}")
        breakpoint()
        
        # Check if output is a tuple or dict
        if isinstance(output, tuple):
            print(f"Output is a tuple with {len(output)} elements")
            for i, elem in enumerate(output):
                if isinstance(elem, torch.Tensor):
                    print(f"  Element {i}: Tensor with shape {elem.shape}")
                else:
                    print(f"  Element {i}: {type(elem)}")
        elif isinstance(output, dict):
            print(f"Output is a dict with keys: {list(output.keys())}")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor with shape {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        elif isinstance(output, torch.Tensor):
            print(f"Output is a single Tensor with shape {output.shape}")
        else:
            print(f"Output: {output}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test collect_feats
    print("\n" + "="*60)
    print("Testing collect_feats method...")
    try:
        feats = model.collect_feats(speech=speech, speech_lengths=speech_lengths)
        print("collect_feats successful!")
        print(f"Output keys: {list(feats.keys())}")
        print(f"  feats shape: {feats['feats'].shape}")
        print(f"  feats_lengths: {feats['feats_lengths']}")
    except Exception as e:
        print(f"Error in collect_feats: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    test_owsm_finetune()
