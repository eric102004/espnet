#!/usr/bin/env python3
"""Test script for Transformers WhisperForConditionalGeneration model."""

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def test_whisper_transformers():
    """Test the WhisperForConditionalGeneration model with dummy data."""
    
    # Model configuration
    model_name = "openai/whisper-base"  # Options: tiny, base, small, medium, large
    
    print(f"Loading WhisperForConditionalGeneration model: {model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Set model to eval mode for testing
    model.eval()
    
    # Create dummy input audio (waveform)
    batch_size = 2
    sample_rate = 16000
    max_seconds = 30
    max_samples = sample_rate * max_seconds

    input_wav = torch.randn(batch_size, max_samples)
    
    # Create dummy decoder input IDs (for teacher forcing during training)
    max_length = 50
    decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, max_length))
    
    print("\nInput shapes:")
    print(f"  input_wav: {input_wav.shape}")
    print(f"  decoder_input_ids: {decoder_input_ids.shape}")
    
    # Test forward pass with input_features only
    print("\n" + "="*60)
    print("Test 1: Forward pass with input_wav only...")
    try:
        with torch.no_grad():
            audio_list = [w.detach().cpu().numpy() for w in input_wav]  # list of (T,)

            processed = processor(
                audio_list,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            output = model(
                input_features=processed.input_features,
                decoder_input_ids=decoder_input_ids,
            )
            breakpoint()
        
        print("Forward pass successful!")
        print(f"\nOutput type: {type(output)}")
        print(f"Output attributes: {dir(output)}")
        
        if hasattr(output, 'logits'):
            print(f"  logits shape: {output.logits.shape}")
        if hasattr(output, 'loss'):
            print(f"  loss: {output.loss}")
        if hasattr(output, 'past_key_values'):
            print(f"  past_key_values: {type(output.past_key_values)}")
            
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass with labels (training mode)
    print("\n" + "="*60)
    print("Test 2: Forward pass with labels (training scenario)...")
    try:
        # Labels are the same as decoder_input_ids but shifted
        labels = decoder_input_ids.clone()
        
        with torch.no_grad():
            processed = processor(
                input_wav,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            output = model(
                input_features=processed.input_features,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
        breakpoint()
        
        print("Forward pass with labels successful!")
        if hasattr(output, 'loss'):
            print(f"  loss: {output.loss.item()}")
        if hasattr(output, 'logits'):
            print(f"  logits shape: {output.logits.shape}")
            
    except Exception as e:
        print(f"Error during forward pass with labels: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test forward pass with decoder_input_ids
    print("\n" + "="*60)
    print("Test 3: Forward pass with decoder_input_ids...")
    try:
        with torch.no_grad():
            processed = processor(
                input_wav,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            output = model(
                input_features=processed.input_features,
                decoder_input_ids=decoder_input_ids,
            )
        breakpoint()
        
        print("Forward pass with decoder_input_ids successful!")
        if hasattr(output, 'logits'):
            print(f"  logits shape: {output.logits.shape}")
            print(f"  logits range: [{output.logits.min().item():.4f}, {output.logits.max().item():.4f}]")
            
    except Exception as e:
        print(f"Error during forward pass with decoder_input_ids: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test generation (inference)
    print("\n" + "="*60)
    print("Test 4: Testing generation (inference)...")
    try:
        # Use smaller input for generation test
        single_input = input_wav[:1]  # Take first sample only
        
        with torch.no_grad():
            processed = processor(
                single_input,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            generated_ids = model.generate(
                inputs=processed.input_features,
                max_length=50,
                num_beams=1,
            )
        
        print("Generation successful!")
        print(f"  generated_ids shape: {generated_ids.shape}")
        print(f"  generated_ids: {generated_ids[0][:20]}...")  # Show first 20 tokens
        
        # Decode the generated IDs
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"  decoded transcription: '{transcription[0]}'")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test encoder
    print("\n" + "="*60)
    print("Test 5: Testing encoder separately...")
    try:
        with torch.no_grad():
            processed = processor(
                input_wav,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            encoder_outputs = model.get_encoder()(processed.input_features)
        
        print("Encoder forward pass successful!")
        if hasattr(encoder_outputs, 'last_hidden_state'):
            print(f"  encoder output shape: {encoder_outputs.last_hidden_state.shape}")
        else:
            print(f"  encoder output type: {type(encoder_outputs)}")
            if isinstance(encoder_outputs, tuple):
                print(f"  encoder output[0] shape: {encoder_outputs[0].shape}")
            
    except Exception as e:
        print(f"Error during encoder forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print model summary
    print("\n" + "="*60)
    print("Model Summary:")
    print(f"  Vocabulary size: {model.config.vocab_size}")
    print(f"  Max source positions: {model.config.max_source_positions}")
    print(f"  Max target positions: {model.config.max_target_positions}")
    print(f"  Encoder layers: {model.config.encoder_layers}")
    print(f"  Decoder layers: {model.config.decoder_layers}")
    print(f"  d_model: {model.config.d_model}")
    print(f"  Encoder attention heads: {model.config.encoder_attention_heads}")
    print(f"  Decoder attention heads: {model.config.decoder_attention_heads}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    test_whisper_transformers()
