SYNTHETIC SIGHT STREAMLIT REPAIR

1. Obtain the final best_resnet50.pth created by
   training_exploration_reinteration_kristine.ipynb.

2. The correct file must have this SHA-256:
   d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c

3. Put it here:
   models/best_resnet50.pth

4. Install:
   python -m pip install -r requirements.txt

5. Verify before launching:
   python verify_checkpoint.py

6. Launch:
   streamlit run app.py

The app intentionally refuses to run with the older epoch-9 checkpoint.
