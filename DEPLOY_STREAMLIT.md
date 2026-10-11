# Deploy on Streamlit Community Cloud (free)

1. Fork/clone the repo to YOUR GitHub account:
   https://github.com/AbderrezzakMrch/Parkinson-s-Disease-Voice-Detector
2. Add these 3 files to the REPO ROOT (next to package.json), commit and push:
   - app.py
   - requirements.txt
   - packages.txt   (installs ffmpeg + libsndfile on the server)
   Keep ml/best_pd_model.pkl where it is (app.py loads it from there).
3. Go to https://share.streamlit.io -> sign in with GitHub -> "Create app".
4. Choose your repo, branch `main`, main file path `app.py`.
5. In Advanced settings choose Python 3.11 or 3.12, then click Deploy.
6. You get a public URL like https://<name>.streamlit.app

Run locally first (optional):
   pip install -r requirements.txt
   streamlit run app.py

Notes
- scikit-learn is pinned to 1.6.1 because the .pkl model was trained with it.
- The React/Vite frontend and FastAPI server are not used on Streamlit;
  app.py reuses the same feature extraction and SVM model directly.
- Login/Signup pages are skipped (their backend does not exist in the repo).
