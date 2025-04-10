# bana-7075-final
Final project for BANA 7075: Machine Learning Design

## About the project
this project...


### Data Pipeline
```
┌────────────┐
│ Ingestion  │ ──> get_data()
└────────────┘
       ↓
┌────────────┐
│ Cleaning   │ ──> clean_data()
└────────────┘
       ↓
┌────────────┐
│ Feature Eng│ ──> create_features()
└────────────┘
       ↓
┌────────────┐
│ Versioning │ ──> version_data()
└────────────┘
       ↓
┌────────────┐
│ Encoding   │ ──> (e.g., one-hot if needed)
└────────────┘
       ↓
┌────────────┐
│ Split Data │ ──> split_data()
└────────────┘
       ↓
┌────────────┐
│ Scaling    │ ──> scale_train_data()
└────────────┘

```