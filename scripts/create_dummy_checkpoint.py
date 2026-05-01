import os
import torch
import sys
# Dummy Checkpoints
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.affective_model import AffectiveModel

def main():
    print("Creating dummy checkpoint...")
    
    try:
        # Needs to match config defaults
        model = AffectiveModel(
            backbone_name="resnet18",
            pretrained=False,
            embedding_dim=256,
            dropout=0.5
        )
        
        checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "best_metric": 0.0,
        }, checkpoint_path)
        
        print(f"Successfully created dummy checkpoint at: {checkpoint_path}")
        print("You can now run inference scripts and the dashboard without training first.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
