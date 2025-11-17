#!/usr/bin/env python3
"""
Test script to verify that FeatureActionExpert and CrossAttentionActionExpert
can be loaded and saved correctly.
"""
import torch
import os
import tempfile
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_expert():
    """Test FeatureActionExpert save/load functionality"""
    logger.info("=" * 80)
    logger.info("Testing FeatureActionExpert")
    logger.info("=" * 80)

    from models.action_patches import FeatureActionExpert

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "feature_expert_test")

        # Create a model
        logger.info("Creating FeatureActionExpert model...")
        model = FeatureActionExpert(
            action_dim=7,
            dynamic_dim=512,  # Smaller for testing
            hidden_dim=256,
            num_layers=3,
            num_heads=8,
            time_horizon=10,
        )

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Save the model
        logger.info(f"Saving model to {model_path}...")
        model.save_pretrained(model_path)

        # Verify config.json was created
        config_path = os.path.join(model_path, "config.json")
        assert os.path.exists(config_path), "Config file not created!"
        logger.info("✓ Config file created")

        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✓ Config loaded: {config}")

        # Verify model.bin was created
        model_bin = os.path.join(model_path, "pytorch_model.bin")
        assert os.path.exists(model_bin), "Model file not created!"
        logger.info("✓ Model file created")

        # Load the model
        logger.info("Loading model from checkpoint...")
        loaded_model = FeatureActionExpert.from_pretrained(model_path, map_location="cpu")
        logger.info(f"✓ Model loaded successfully")

        # Compare parameters
        logger.info("Comparing model parameters...")
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
            assert torch.allclose(param1, param2, atol=1e-6), f"Parameter {name1} doesn't match!"
        logger.info("✓ All parameters match!")

        # Test forward pass
        logger.info("Testing forward pass...")
        batch_size, seq_len, action_dim = 2, 10, 7
        visual_features = torch.randn(batch_size, 512)
        reward_features = torch.randn(batch_size, 10, 512)
        target_actions = torch.randn(batch_size, seq_len, action_dim)

        reward_sampling_results = {
            'last_hidden_states': torch.randn(batch_size, 100, 512),
            'last_image_token_pos': torch.randint(0, 100, (batch_size, 1)),
            'critical_segments': torch.randn(batch_size, 1, 11, 512)
        }

        with torch.no_grad():
            loss_dict1 = model.compute_flow_loss(
                reward_sampling_results=reward_sampling_results,
                target_actions=target_actions
            )
            loss_dict2 = loaded_model.compute_flow_loss(
                reward_sampling_results=reward_sampling_results,
                target_actions=target_actions
            )

        assert torch.allclose(loss_dict1['loss'], loss_dict2['loss'], atol=1e-6), "Loss values don't match!"
        logger.info(f"✓ Forward pass test passed! Loss: {loss_dict1['loss'].item():.6f}")

    logger.info("\n✓ FeatureActionExpert test PASSED!\n")


def test_cross_attention_expert():
    """Test CrossAttentionActionExpert save/load functionality"""
    logger.info("=" * 80)
    logger.info("Testing CrossAttentionActionExpert")
    logger.info("=" * 80)

    from models.action_patches import CrossAttentionActionExpert

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "cross_attention_expert_test")

        # Create a model
        logger.info("Creating CrossAttentionActionExpert model...")
        model = CrossAttentionActionExpert(
            action_dim=7,
            dynamic_dim=512,
            hidden_dim=256,
            cross_attention_dim=512,
            num_layers=3,
            num_heads=8,
            time_horizon=10,
        )

        logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

        # Save the model
        logger.info(f"Saving model to {model_path}...")
        model.save_pretrained(model_path)

        # Verify config.json was created
        config_path = os.path.join(model_path, "config.json")
        assert os.path.exists(config_path), "Config file not created!"
        logger.info("✓ Config file created")

        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✓ Config loaded: {config}")

        # Verify model.bin was created
        model_bin = os.path.join(model_path, "pytorch_model.bin")
        assert os.path.exists(model_bin), "Model file not created!"
        logger.info("✓ Model file created")

        # Load the model
        logger.info("Loading model from checkpoint...")
        loaded_model = CrossAttentionActionExpert.from_pretrained(model_path, map_location="cpu")
        logger.info(f"✓ Model loaded successfully")

        # Compare parameters
        logger.info("Comparing model parameters...")
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
            assert torch.allclose(param1, param2, atol=1e-6), f"Parameter {name1} doesn't match!"
        logger.info("✓ All parameters match!")

        # Test forward pass
        logger.info("Testing forward pass...")
        batch_size, seq_len, action_dim = 2, 10, 7
        visual_features = torch.randn(batch_size, 512)
        reward_features = torch.randn(batch_size, 512)
        dynamic_hidden_states = torch.randn(batch_size, 50, 512)
        target_actions = torch.randn(batch_size, seq_len, action_dim)

        with torch.no_grad():
            loss_dict1 = model.compute_flow_loss(
                dynamic_hidden_states=dynamic_hidden_states,
                visual_features=visual_features,
                reward_features=reward_features,
                target_actions=target_actions
            )
            loss_dict2 = loaded_model.compute_flow_loss(
                dynamic_hidden_states=dynamic_hidden_states,
                visual_features=visual_features,
                reward_features=reward_features,
                target_actions=target_actions
            )

        assert torch.allclose(loss_dict1['loss'], loss_dict2['loss'], atol=1e-6), "Loss values don't match!"
        logger.info(f"✓ Forward pass test passed! Loss: {loss_dict1['loss'].item():.6f}")

    logger.info("\n✓ CrossAttentionActionExpert test PASSED!\n")


def test_create_action_expert():
    """Test create_action_expert function with loading"""
    logger.info("=" * 80)
    logger.info("Testing create_action_expert function")
    logger.info("=" * 80)

    from models.action_patches import create_action_expert, ActionExpertConfig, ExpertType

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "created_expert")

        # First create and save a model
        logger.info("Creating and saving FeatureActionExpert...")
        config = ActionExpertConfig(
            action_dim=7,
            dynamic_dim=512,
            hidden_dim=256,
            num_layers=3,
            num_heads=8,
            time_horizon=10,
            expert_type=ExpertType.FEATURE_BASED
        )

        model1 = create_action_expert(config)
        model1.save_pretrained(model_path)
        logger.info(f"✓ Model saved to {model_path}")

        # Now load it using create_action_expert
        logger.info("Loading model using create_action_expert...")
        model2 = create_action_expert(config, model_path=model_path, map_location="cpu")
        logger.info(f"✓ Model loaded successfully")

        # Compare parameters
        logger.info("Comparing model parameters...")
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"
            assert torch.allclose(param1, param2, atol=1e-6), f"Parameter {name1} doesn't match!"
        logger.info("✓ All parameters match!")

    logger.info("\n✓ create_action_expert test PASSED!\n")


if __name__ == "__main__":
    try:
        test_feature_expert()
        test_cross_attention_expert()
        test_create_action_expert()

        logger.info("=" * 80)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
