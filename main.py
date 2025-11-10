from src.pipeline.train_pipeline import TrainPipeline


if __name__ == "__main__":
    try:
        pipeline = TrainPipeline()
        pipeline.start_training_model()
    except Exception as e:
        print(f"Pipeline failed: {e}")