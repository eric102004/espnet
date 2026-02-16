#!/usr/bin/env python3
"""Test script for WhisperFinetune model."""

import torch
from model import WhisperFinetune


def test_whisper_finetune():
    """Test the WhisperFinetune model with dummy data."""
    
    # Model configuration
    model_tag = "openai/whisper-tiny"  # Use tiny for faster testing
    
    print(f"Loading WhisperFinetune model: {model_tag}...")
    model = WhisperFinetune(model_tag)
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model.model)}")
    print(f"Processor type: {type(model.processor)}")
    
    # Set model to training mode for testing
    model.train()
    
    # Create dummy input data
    batch_size = 2
    # Speech inputs: mel-spectrogram features (B, T, D)
    # where D=80 is the number of mel bands
    # T is the time dimension
    mel_dim = 80
    max_frames = 1000
    
    speech_len_1 = max_frames
    speech_len_2 = max_frames // 2
    speech = torch.randn(batch_size, max_frames, mel_dim)
    speech_lengths = torch.tensor([speech_len_1, speech_len_2])
    
    # Text inputs (token IDs) - whisper vocab size is 51865
    vocab_size = model.model.config.vocab_size
    max_text_length = 50
    
    text = torch.randint(0, vocab_size, (batch_size, max_text_length))
    text_lengths = torch.tensor([max_text_length, max_text_length // 2])
    
    # CTC text inputs
    text_ctc = torch.randint(0, vocab_size, (batch_size, max_text_length))
    text_ctc_lengths = torch.tensor([max_text_length, max_text_length // 2])
    
    # Previous text inputs (for teacher forcing)
    text_prev = torch.randint(0, vocab_size, (batch_size, max_text_length))
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
    
    # Test forward pass in training mode
    print("\n" + "="*60)
    print("Test 1: Forward pass in training mode...")
    try:
        loss, stats, weight = model.forward(
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
        print(f"\nOutput:")
        print(f"  loss: {loss.item():.4f}")
        print(f"  weight (batch_size): {weight}")
        print(f"\nStats:")
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: None (only computed in eval mode)")
                
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass in eval mode
    print("\n" + "="*60)
    print("Test 2: Forward pass in eval mode...")
    try:
        model.eval()
        
        with torch.no_grad():
            loss, stats, weight = model.forward(
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
        print(f"\nOutput:")
        print(f"  loss: {loss.item():.4f}")
        print(f"  weight (batch_size): {weight}")
        print(f"\nStats (with CER/WER):")
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: None")
                
    except Exception as e:
        print(f"Error during eval forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test collect_feats
    print("\n" + "="*60)
    print("Test 3: Testing collect_feats method...")
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
    
    # Test with different batch sizes
    print("\n" + "="*60)
    print("Test 4: Testing with single sample (batch_size=1)...")
    try:
        single_speech = speech[:1]
        single_speech_lengths = speech_lengths[:1]
        single_text = text[:1]
        single_text_lengths = text_lengths[:1]
        single_text_ctc = text_ctc[:1]
        single_text_ctc_lengths = text_ctc_lengths[:1]
        single_text_prev = text_prev[:1]
        single_text_prev_lengths = text_prev_lengths[:1]
        
        with torch.no_grad():
            loss, stats, weight = model.forward(
                speech=single_speech,
                speech_lengths=single_speech_lengths,
                text=single_text,
                text_lengths=single_text_lengths,
                text_ctc=single_text_ctc,
                text_ctc_lengths=single_text_ctc_lengths,
                text_prev=single_text_prev,
                text_prev_lengths=single_text_prev_lengths,
            )
        
        print("Single sample forward pass successful!")
        print(f"  loss: {loss.item():.4f}")
        print(f"  weight: {weight}")
        
    except Exception as e:
        print(f"Error with single sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print model summary
    print("\n" + "="*60)
    print("Model Summary:")
    print(f"  Model config vocab_size: {model.model.config.vocab_size}")
    print(f"  Max source positions: {model.model.config.max_source_positions}")
    print(f"  Max target positions: {model.model.config.max_target_positions}")
    print(f"  Encoder layers: {model.model.config.encoder_layers}")
    print(f"  Decoder layers: {model.model.config.decoder_layers}")
    print(f"  d_model: {model.model.config.d_model}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    test_whisper_finetune()
