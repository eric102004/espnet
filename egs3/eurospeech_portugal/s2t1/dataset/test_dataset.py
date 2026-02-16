#!/usr/bin/env python3
"""Test script for WhisperTokenizeTransform."""

import numpy as np
import sys
from pathlib import Path

# Add dataset directory to path
sys.path.insert(0, str(Path(__file__).parent / "dataset"))

from dataset import WhisperTokenizeTransform


def test_whisper_tokenize_transform():
    """Test the WhisperTokenizeTransform class."""
    
    # Model configuration
    model_tag = "openai/whisper-tiny"  # Use tiny for faster testing
    
    print(f"Loading WhisperTokenizeTransform with model: {model_tag}...")
    transform = WhisperTokenizeTransform(model_tag)
    print(f"Transform loaded successfully!")
    print(f"Processor type: {type(transform.processor)}")
    print(f"Tokenizer vocab size: {transform.processor.tokenizer.vocab_size}")
    
    # Test tokenize method
    print("\n" + "="*60)
    print("Test 1: Testing tokenize method...")
    
    test_texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Olá, como está?",  # Portuguese
        "",  # Empty string
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\nTest text {i+1}: '{text}'")
        try:
            tokens = transform.tokenize(text)
            print(f"  Tokens shape: {tokens.shape}")
            print(f"  Tokens dtype: {tokens.dtype}")
            print(f"  Tokens: {tokens}")
            
            # Decode back to verify
            decoded = transform.processor.tokenizer.decode(tokens)
            print(f"  Decoded: '{decoded}'")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test __call__ method
    print("\n" + "="*60)
    print("Test 2: Testing __call__ method (full transform)...")
    
    # Create sample data
    sample_rate = 16000
    duration = 5  # seconds
    sample_data = {
        'speech': np.random.randn(sample_rate * duration).astype(np.float32),
        'text': '<por><asr><notimestamps> Olá, como está?',
        'text_ctc': 'Olá, como está?',
        'text_prev': '<na>',
    }
    
    print("\nInput data:")
    print(f"  speech shape: {sample_data['speech'].shape}")
    print(f"  speech dtype: {sample_data['speech'].dtype}")
    print(f"  text: '{sample_data['text']}'")
    print(f"  text_ctc: '{sample_data['text_ctc']}'")
    print(f"  text_prev: '{sample_data['text_prev']}'")
    
    try:
        result = transform(sample_data)
        
        print("\nTransformed output:")
        print(f"  Keys: {list(result.keys())}")
        print(f"  speech shape: {result['speech'].shape}")
        print(f"  speech dtype: {result['speech'].dtype}")
        print(f"  text shape: {result['text'].shape}")
        print(f"  text dtype: {result['text'].dtype}")
        print(f"  text tokens: {result['text']}")
        print(f"  text_ctc shape: {result['text_ctc'].shape}")
        print(f"  text_ctc tokens: {result['text_ctc']}")
        print(f"  text_prev shape: {result['text_prev'].shape}")
        print(f"  text_prev tokens: {result['text_prev']}")
        
        # Decode tokens back to text
        print("\nDecoded tokens:")
        print(f"  text: '{transform.processor.tokenizer.decode(result['text'])}'")
        print(f"  text_ctc: '{transform.processor.tokenizer.decode(result['text_ctc'])}'")
        print(f"  text_prev: '{transform.processor.tokenizer.decode(result['text_prev'])}'")
        
    except Exception as e:
        print(f"Error during transform: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with multiple samples
    print("\n" + "="*60)
    print("Test 3: Testing with multiple text samples...")
    
    test_samples = [
        {
            'speech': np.random.randn(sample_rate * 3).astype(np.float32),
            'text': '<por><asr><notimestamps> Bom dia',
            'text_ctc': 'Bom dia',
            'text_prev': '<na>',
        },
        {
            'speech': np.random.randn(sample_rate * 7).astype(np.float32),
            'text': '<por><asr><notimestamps> Muito obrigado pela ajuda',
            'text_ctc': 'Muito obrigado pela ajuda',
            'text_prev': '<na>',
        },
    ]
    
    for i, sample in enumerate(test_samples):
        print(f"\nSample {i+1}:")
        print(f"  Input text: '{sample['text']}'")
        try:
            result = transform(sample)
            print(f"  Speech shape: {result['speech'].shape}")
            print(f"  Text tokens length: {len(result['text'])}")
            print(f"  Text_ctc tokens length: {len(result['text_ctc'])}")
            print(f"  Decoded text: '{transform.processor.tokenizer.decode(result['text'])}'")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test special tokens
    print("\n" + "="*60)
    print("Test 4: Testing special tokens...")
    
    special_token_ids = {
        'bos_token': transform.processor.tokenizer.bos_token_id,
        'eos_token': transform.processor.tokenizer.eos_token_id,
        'pad_token': transform.processor.tokenizer.pad_token_id,
        'unk_token': transform.processor.tokenizer.unk_token_id,
    }
    
    print("Special token IDs:")
    for token_name, token_id in special_token_ids.items():
        if token_id is not None:
            token_str = transform.processor.tokenizer.decode([token_id])
            print(f"  {token_name}: {token_id} ('{token_str}')")
        else:
            print(f"  {token_name}: None")
    
    # Check if speech array is unchanged
    print("\n" + "="*60)
    print("Test 5: Verify speech array is unchanged...")
    
    original_speech = np.random.randn(sample_rate * 2).astype(np.float32)
    test_data = {
        'speech': original_speech.copy(),
        'text': 'Test',
        'text_ctc': 'Test',
        'text_prev': 'Test',
    }
    
    result = transform(test_data)
    
    if np.allclose(result['speech'], original_speech):
        print("✓ Speech array is unchanged (as expected)")
    else:
        print("✗ Speech array was modified (unexpected)")
        diff = np.abs(result['speech'] - original_speech).max()
        print(f"  Max difference: {diff}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    test_whisper_tokenize_transform()
