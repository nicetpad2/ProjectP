
def validate_enterprise_data_quality(X, y, temporal_col=None):
    """ตรวจสอบคุณภาพข้อมูลระดับองค์กร"""
    
    results = {
        'overall_quality': 0.0,
        'issues_found': [],
        'recommendations': [],
        'enterprise_ready': False
    }
    
    # 1. ตรวจสอบ Missing Values
    missing_pct = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
    if missing_pct > 0.01:  # ไม่เกิน 1%
        results['issues_found'].append(f"High missing values: {missing_pct:.2%}")
        results['recommendations'].append("Implement robust missing value handling")
    
    # 2. ตรวจสอบ Outliers
    from scipy import stats
    outlier_count = 0
    for col in X.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(X[col].dropna()))
        outliers = (z_scores > 3).sum()
        outlier_count += outliers
    
    outlier_pct = outlier_count / (X.shape[0] * X.shape[1])
    if outlier_pct > 0.05:  # ไม่เกิน 5%
        results['issues_found'].append(f"High outlier percentage: {outlier_pct:.2%}")
        results['recommendations'].append("Implement outlier detection and treatment")
    
    # 3. ตรวจสอบ Data Leakage (High correlation with target)
    correlations = []
    for col in X.select_dtypes(include=[np.number]).columns:
        corr = abs(np.corrcoef(X[col].fillna(0), y)[0, 1])
        if corr > 0.95:  # สงสัยว่าเป็น data leakage
            correlations.append((col, corr))
    
    if correlations:
        results['issues_found'].append(f"Suspicious high correlations: {len(correlations)} features")
        results['recommendations'].append("Investigate potential data leakage")
    
    # 4. ตรวจสอบ Feature Stability
    if X.shape[1] > 0:
        feature_stability = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() > 0:
                cv = X[col].std() / abs(X[col].mean()) if X[col].mean() != 0 else float('inf')
                feature_stability.append(cv)
        
        avg_stability = np.mean(feature_stability) if feature_stability else 0
        if avg_stability > 2.0:  # CV > 2 indicates unstable features
            results['issues_found'].append(f"Unstable features detected: avg CV = {avg_stability:.2f}")
            results['recommendations'].append("Apply feature scaling and normalization")
    
    # 5. ตรวจสอบ Temporal Consistency (ถ้ามี temporal column)
    if temporal_col is not None and temporal_col in X.columns:
        if not pd.api.types.is_datetime64_any_dtype(X[temporal_col]):
            results['issues_found'].append("Temporal column is not datetime type")
            results['recommendations'].append("Convert temporal column to datetime")
        
        # ตรวจสอบ gaps ในข้อมูล
        time_series = pd.to_datetime(X[temporal_col]).sort_values()
        time_gaps = time_series.diff().dropna()
        if time_gaps.std() > time_gaps.mean():
            results['issues_found'].append("Irregular time intervals detected")
            results['recommendations'].append("Handle irregular time series patterns")
    
    # คำนวณคะแนนรวม
    max_issues = 5  # จำนวน issues สูงสุดที่ตรวจสอบ
    issues_penalty = len(results['issues_found']) / max_issues
    results['overall_quality'] = max(0.0, 1.0 - issues_penalty)
    
    # Enterprise readiness
    results['enterprise_ready'] = (
        results['overall_quality'] >= 0.95 and
        len(results['issues_found']) <= 1
    )
    
    return results

def clean_enterprise_data(X, y, cleaning_config=None):
    """ทำความสะอาดข้อมูลตามมาตรฐานองค์กร"""
    
    if cleaning_config is None:
        cleaning_config = {
            'handle_missing': True,
            'remove_outliers': True,
            'scale_features': True,
            'remove_correlated_features': True,
            'correlation_threshold': 0.95
        }
    
    X_clean = X.copy()
    
    # 1. Handle missing values
    if cleaning_config['handle_missing']:
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            if X_clean[col].isnull().sum() > 0:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
    
    # 2. Remove outliers
    if cleaning_config['remove_outliers']:
        from scipy import stats
        mask = np.ones(len(X_clean), dtype=bool)
        
        for col in X_clean.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(X_clean[col]))
            mask &= (z_scores < 3)  # Keep only values within 3 standard deviations
        
        X_clean = X_clean[mask]
        y_clean = y[mask] if hasattr(y, '__getitem__') else y
    else:
        y_clean = y
    
    # 3. Scale features
    if cleaning_config['scale_features']:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean[numeric_cols] = scaler.fit_transform(X_clean[numeric_cols])
    
    # 4. Remove highly correlated features
    if cleaning_config['remove_correlated_features']:
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = X_clean[numeric_cols].corr().abs()
            
            # Find pairs of highly correlated features
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > cleaning_config['correlation_threshold'])]
            
            X_clean = X_clean.drop(columns=to_drop)
    
    return X_clean, y_clean
