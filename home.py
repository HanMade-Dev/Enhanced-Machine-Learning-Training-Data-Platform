import streamlit as st

def home_page():
    st.title("🤖 Machine Learning Training Data Platform")
    st.markdown("### *Enhanced Edition dengan Advanced Features*")
    st.markdown("###### *By CoDev Labs*")
    st.markdown("---")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; margin-bottom: 30px; border-left: 5px solid #1f77b4;">
        <h3>🎯 Selamat Datang di Platform ML Enhanced</h3>
        <p style="font-size: 16px; margin-bottom: 15px;">
            Platform machine learning canggih yang dilengkapi dengan fitur-fitur terdepan untuk training model dan klasifikasi data. 
            Dengan interface yang intuitif dan teknologi terkini, Anda dapat membangun model ML berkualitas tinggi tanpa coding yang rumit.
        </p>
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px;">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">✨ Fitur Terbaru:</h4>
            <ul style="margin-bottom: 0; color: #34495e;">
                <li><strong>Advanced Data Upload:</strong> Auto-detection, preview,  dan validasi data</li>
                <li><strong>Smart Preprocessing:</strong> Pipeline yang dapat dikonfigurasi berkali-kali</li>
                <li><strong>Intelligent Feature Engineering:</strong> Otomatis dan manual selection</li>
                <li><strong>Model Training:</strong> Hyperparameter tuning dan comparison</li>
                <li><strong>Enhanced Evaluation:</strong> Metrics lengkap dan visualisasi interaktif</li>
                <li><strong>Direct Model Download:</strong> Save dan download model langsung</li>
                <li><strong>String Label Support:</strong> Klasifikasi dengan label string</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🚀 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 12px; border: 2px solid #e1e5e9; margin-bottom: 15px; transition: all 0.3s ease;">
            <h3 style="color: black; margin-bottom: 15px;">📊 Advanced Training Platform</h3>
            <ul style="margin-bottom: 15px;">
                <li><strong>Smart Data Upload:</strong> Multi-format support dengan auto-detection</li>
                <li><strong>Configurable Preprocessing:</strong> Pipeline yang dapat diulang dan divalidasi</li>
                <li><strong>Intelligent Feature Selection:</strong> Manual dan automated methods</li>
                <li><strong>Advanced Model Training:</strong> 10+ algorithms dengan hyperparameter tuning</li>
                <li><strong>Comprehensive Evaluation:</strong> Metrics lengkap dan visualisasi</li>
                <li><strong>Direct Download:</strong> Model dan report siap deploy</li>
            </ul>
            <p style="margin-bottom: 0; font-style: italic;">Perfect untuk data scientists dan ML engineers</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 START TRAINING", type="primary", use_container_width=True):
            st.session_state.current_page = 'training'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 12px; border: 2px solid #e1e5e9; margin-bottom: 15px; transition: all 0.3s ease;">
            <h3 style="color: black; margin-bottom: 15px;">🎯 Enhanced Classification</h3>
            <ul style="margin-bottom: 15px;">
                <li><strong>Smart Model Loading:</strong> Support untuk berbagai format model</li>
                <li><strong>String Label Support:</strong> Klasifikasi dengan label string asli</li>
                <li><strong>Flexible Input:</strong> Manual input dan batch file processing</li>
                <li><strong>Advanced Predictions:</strong> Probability scores dan confidence metrics</li>
                <li><strong>Interactive Visualizations:</strong> Charts dan graphs untuk hasil</li>
                <li><strong>Export Results:</strong> Download hasil dalam format CSV</li>
            </ul>
            <p style="margin-bottom: 0; font-style: italic;">Perfect untuk production deployment dan testing</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 START CLASSIFICATION", type="primary", use_container_width=True):
            st.session_state.current_page = 'classification'
            st.rerun()
    
    st.markdown("---")
    st.subheader("🔀 Enhanced ML Workflow")
    
    workflow_steps = [
        {
            "icon": "📤",
            "title": "Smart Data Upload",
            "description": "Upload CSV/Excel dengan auto-detection format, preview data, dan validasi kualitas",
            "features": ["Multi-file support", "Auto delimiter detection", "Data quality validation", "Column selection"]
        },
        {
            "icon": "⚙️",
            "title": "Configurable Preprocessing", 
            "description": "Pipeline preprocessing yang dapat dikonfigurasi berkali-kali dengan preview dan validasi",
            "features": ["Missing values handling", "Outlier detection", "Feature engineering", "Data normalization"]
        },
        {
            "icon": "🎯",
            "title": "Intelligent Feature Selection",
            "description": "Kombinasi manual selection dan automated feature selection methods",
            "features": ["Manual labeling", "Correlation analysis", "Statistical tests", "Feature importance"]
        },
        {
            "icon": "🤖",
            "title": "Advanced Model Training",
            "description": "Training dengan hyperparameter tuning, model comparison, dan cross-validation",
            "features": ["10+ algorithms", "Hyperparameter tuning", "Model comparison", "Cross-validation"]
        },
        {
            "icon": "📈",
            "title": "Comprehensive Evaluation",
            "description": "Evaluasi lengkap dengan metrics, visualisasi interaktif, dan detailed reports",
            "features": ["Performance metrics", "Interactive charts", "Feature analysis", "HTML reports"]
        },
        {
            "icon": "💾",
            "title": "Direct Download & Deploy",
            "description": "Download model langsung dengan LabelEncoder support dan deployment instructions",
            "features": ["Direct download", "String label support", "Deployment code", "Usage examples"]
        }
    ]
    
    for i in range(0, len(workflow_steps), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            step = workflow_steps[i]
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 12px; border: 2px solid #e1e5e9; margin-bottom: 15px; transition: all 0.3s ease;">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 32px; margin-right: 15px;">{step['icon']}</div>
                    <h4 style="margin: 0; color: #2c3e50;">{step['title']}</h4>
                </div>
                <p style="color: #5a6c7d; margin-bottom: 15px; line-height: 1.6;">{step['description']}</p>
                <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px;">
                    <strong style="color: #495057;">Key Features:</strong>
                    <ul style="margin: 8px 0 0 0; color: #6c757d;">
                        {''.join([f'<li>{feature}</li>' for feature in step['features']])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if i + 1 < len(workflow_steps):
            with col2:
                step = workflow_steps[i + 1]
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 12px; border: 2px solid #e1e5e9; margin-bottom: 15px; transition: all 0.3s ease;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <div style="font-size: 32px; margin-right: 15px;">{step['icon']}</div>
                        <h4 style="margin: 0; color: #2c3e50;">{step['title']}</h4>
                    </div>
                    <p style="color: #5a6c7d; margin-bottom: 15px; line-height: 1.6;">{step['description']}</p>
                    <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px;">
                        <strong style="color: #495057;">Key Features:</strong>
                        <ul style="margin: 8px 0 0 0; color: #6c757d;">
                            {''.join([f'<li>{feature}</li>' for feature in step['features']])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔧 Technical Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Supported Data Formats:**
        - CSV (auto-delimiter detection)
        - Excel (.xlsx, .xls)
        - Multi-file upload
        - Indonesian number format support
        - Automatic data type detection
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning Algorithms:**
        - Random Forest (Classifier/Regressor)
        - Gradient Boosting (Classifier/Regressor)
        - Decision Tree (Classifier/Regressor)
        - SVM (Classifier/Regressor)
        - Logistic/Linear Regression
        - K-Nearest Neighbors
        - Naive Bayes
        - Clustering (K-Means, DBSCAN, Hierarchical)
        - PCA Dimensionality Reduction
        """)
    
    with col3:
        st.markdown("""
        **📈 Advanced Features:**
        - Hyperparameter tuning (Grid/Random Search)
        - Cross-validation
        - Model comparison
        - Feature importance analysis
        - Interactive visualizations
        - Comprehensive evaluation metrics
        - HTML report generation
        - Direct model download
        - String label support
        - Deployment instructions
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-top: 30px; color: white;">
        <h1>👨‍💻 </h1>
        <h4 style="color: white; margin-bottom: 15px;"><strong>Dikembangkan Oleh HanDev</strong></h4>
        <a href="https://github.com/HanMade-Dev" target="_blank" style="
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 0.8rem 2rem;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
            border: 2px solid rgba(255,255,255,0.3);
        ">
            🔗 Visit GitHub Repository
        </a>
        <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 15px; margin-top: 20px;">
            <p style="color: #e8e8e8; font-size: 14px; margin: 0; line-height: 1.6;">
                <strong>Enhanced ML Platform V2</strong> | Powered by Streamlit & Scikit-learn<br>
                <em>Advanced features for modern ML workflows</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
