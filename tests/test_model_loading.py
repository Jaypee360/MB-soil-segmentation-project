import mlflow
import mlflow.pyfunc
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

def minimal_model_test():
    """
    Minimal test focusing just on model loading without sklearn dependencies
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print("🔍 Minimal Model Loading Test...\n")
    
    try:
        # Try direct XGBoost loading (bypasses some sklearn issues)
        run_id = "4c0597f6a4e24447853dd51fb297efc2" # magnificent-slug-957
        model_uri = f"runs:/{run_id}/soil_capability_classifier"
        
        print(f"Loading: {model_uri}")
        
        # Load as generic MLflow model first
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        
        print("✅ SUCCESS: Model loaded as PyFunc!")
        print(f"   Model type: {type(pyfunc_model)}")
        
        # Test with dummy prediction
        dummy_data = np.random.rand(1, 30)  # 30 features
        prediction = pyfunc_model.predict(dummy_data)
        
        print(f"✅ Prediction works: {prediction}")
        print(f"\n🎯 Model URI for production: {model_uri}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    success = minimal_model_test()
    if success:
        print("\n✅ Your model is loadable for production!")
    else:
        print("\n❌ Model loading needs troubleshooting")