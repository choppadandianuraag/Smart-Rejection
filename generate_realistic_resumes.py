"""
Generate 50 UNIQUE test resumes as PDF files.
Each resume has randomized projects, skills, experience, education.
- 30 Data Science resumes (ranging from excellent to poor quality)
- 20 Other job role resumes
"""

import random
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_CENTER


OUTPUT_DIR = Path(__file__).parent / "test_resumes" / "pdfs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Name', parent=styles['Heading1'], fontSize=16, alignment=TA_CENTER, spaceAfter=4))
    styles.add(ParagraphStyle(name='Contact', parent=styles['Normal'], fontSize=9, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], fontSize=11, spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubHeader', parent=styles['Normal'], fontSize=10, fontName='Helvetica-Bold', spaceAfter=2))
    styles.add(ParagraphStyle(name='SubHeaderItalic', parent=styles['Normal'], fontSize=9, fontName='Helvetica-Oblique', spaceAfter=3))
    styles.add(ParagraphStyle(name='BulletItem', parent=styles['Normal'], fontSize=9, leftIndent=15, spaceAfter=2))
    styles.add(ParagraphStyle(name='ResumeBody', parent=styles['Normal'], fontSize=9, spaceAfter=4))
    styles.add(ParagraphStyle(name='SkillCategory', parent=styles['Normal'], fontSize=9, spaceAfter=2))
    return styles


def create_resume_pdf(data: dict, filepath: Path):
    doc = SimpleDocTemplate(str(filepath), pagesize=A4, rightMargin=0.6*inch, leftMargin=0.6*inch, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = create_styles()
    story = []
    
    # Header
    story.append(Paragraph(data["name"].upper(), styles['Name']))
    contact = f"{data['location']} | {data['phone']} | {data['email']}"
    story.append(Paragraph(contact, styles['Contact']))
    if data.get('linkedin') or data.get('github'):
        links = f"LinkedIn: {data.get('linkedin', 'N/A')} | GitHub: {data.get('github', 'N/A')}"
        story.append(Paragraph(links, styles['Contact']))
    
    # Professional Summary
    story.append(Paragraph("PROFESSIONAL SUMMARY", styles['SectionTitle']))
    story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
    story.append(Spacer(1, 4))
    story.append(Paragraph(data["summary"], styles['ResumeBody']))
    
    # Education
    story.append(Paragraph("EDUCATION", styles['SectionTitle']))
    story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
    story.append(Spacer(1, 4))
    for edu in data["education"]:
        story.append(Paragraph(f"<b>{edu['institution']}</b> <i>({edu['year']})</i>", styles['SubHeader']))
        story.append(Paragraph(f"{edu['degree']} | {edu['score']}", styles['ResumeBody']))
    
    # Skills
    story.append(Paragraph("SKILLS", styles['SectionTitle']))
    story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
    story.append(Spacer(1, 4))
    for category, skills in data["skills"].items():
        story.append(Paragraph(f"<b>{category}:</b> {', '.join(skills)}", styles['SkillCategory']))
    
    # Experience (if any)
    if data.get("experience"):
        story.append(Paragraph("EXPERIENCE", styles['SectionTitle']))
        story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
        story.append(Spacer(1, 4))
        for exp in data["experience"]:
            story.append(Paragraph(f"<b>{exp['title']}</b> | {exp['company']} <i>({exp['duration']})</i>", styles['SubHeader']))
            for bullet in exp["bullets"]:
                story.append(Paragraph(f"* {bullet}", styles['BulletItem']))
            story.append(Spacer(1, 4))
    
    # Projects
    story.append(Paragraph("PROJECTS", styles['SectionTitle']))
    story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
    story.append(Spacer(1, 4))
    for proj in data["projects"]:
        story.append(Paragraph(f"<b>{proj['name']}</b> <i>({proj['date']})</i>", styles['SubHeader']))
        story.append(Paragraph(f"<i>Tools: {proj['tools']}</i>", styles['SubHeaderItalic']))
        story.append(Paragraph(proj["description"], styles['BulletItem']))
        story.append(Spacer(1, 3))
    
    # Certifications
    if data.get("certifications"):
        story.append(Paragraph("CERTIFICATIONS", styles['SectionTitle']))
        story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
        story.append(Spacer(1, 4))
        for cert in data["certifications"]:
            story.append(Paragraph(f"* {cert}", styles['BulletItem']))
    
    # Achievements (if any)
    if data.get("achievements"):
        story.append(Paragraph("ACHIEVEMENTS", styles['SectionTitle']))
        story.append(HRFlowable(width="100%", thickness=0.5, color="black"))
        story.append(Spacer(1, 4))
        for ach in data["achievements"]:
            story.append(Paragraph(f"* {ach}", styles['BulletItem']))
    
    doc.build(story)


# ============================================================================
# RAW DATA POOLS FOR RANDOMIZATION
# ============================================================================

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
               "Ananya", "Diya", "Myra", "Sara", "Aanya", "Aadhya", "Isha", "Kavya", "Avni", "Prisha",
               "Rohan", "Arnav", "Dhruv", "Kabir", "Shaurya", "Atharv", "Advait", "Aarush", "Virat", "Rudra",
               "Saanvi", "Anika", "Pari", "Navya", "Angel", "Riya", "Kiara", "Mira", "Anvi", "Zara",
               "Yash", "Kartik", "Mohit", "Rahul", "Amit", "Sumit", "Nikhil", "Varun", "Karan", "Gaurav"]

LAST_NAMES = ["Sharma", "Verma", "Patel", "Singh", "Gupta", "Reddy", "Kumar", "Nair", "Iyer", "Rao",
              "Joshi", "Malhotra", "Chopra", "Kapoor", "Mehta", "Agarwal", "Tiwari", "Pandey", "Mishra", "Saxena",
              "Desai", "Shah", "Kulkarni", "Menon", "Pillai", "Hegde", "Bhat", "Venkatesh", "Krishnan", "Bansal"]

CITIES = ["Hyderabad, Telangana", "Bangalore, Karnataka", "Mumbai, Maharashtra", "Delhi NCR", 
          "Chennai, Tamil Nadu", "Pune, Maharashtra", "Kolkata, West Bengal", "Ahmedabad, Gujarat",
          "Noida, Uttar Pradesh", "Gurgaon, Haryana"]

# Tiered Colleges
COLLEGES_TIER1 = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur", "IISc Bangalore", 
                  "BITS Pilani", "NIT Trichy", "IIIT Hyderabad", "IIT Roorkee", "IIT Guwahati"]
COLLEGES_TIER2 = ["VIT Vellore", "SRM Chennai", "Manipal Institute of Technology", "PESIT Bangalore", 
                  "RV College of Engineering", "BMS College of Engineering", "NIT Warangal", "NIT Surathkal",
                  "IIIT Bangalore", "DTU Delhi", "NSIT Delhi", "COEP Pune"]
COLLEGES_TIER3 = ["VNR VJIET Hyderabad", "CBIT Hyderabad", "Vasavi College of Engineering", "MVSR Engineering College",
                  "CVR College of Engineering", "GRIET Hyderabad", "Sathyabama Chennai", "Amity University",
                  "LPU Punjab", "Chandigarh University", "Shiv Nadar University"]
COLLEGES_TIER4 = ["Regional Engineering College", "State University of Technology", "City Institute of Technology",
                  "Private Engineering Academy", "District Polytechnic College"]

# Schools for 12th/10th
SCHOOLS = ["Delhi Public School", "Kendriya Vidyalaya", "DAV Public School", "St. Xavier's High School",
           "Ryan International School", "Army Public School", "Narayana Junior College", "Sri Chaitanya",
           "FIITJEE Junior College", "Resonance Academy", "Allen Career Institute"]

BOARDS = ["CBSE Board", "ICSE Board", "State Board", "IB Board"]

# ============================================================================
# DS PROJECT TEMPLATES - Mix and match components
# ============================================================================

DS_PROJECT_DOMAINS = [
    "Healthcare", "Finance", "E-commerce", "Social Media", "Manufacturing", "Retail", 
    "Transportation", "Energy", "Telecom", "Education", "Agriculture", "Real Estate"
]

DS_PROJECT_TYPES = {
    "classification": {
        "names": [
            "{domain} Customer Churn Prediction System",
            "{domain} Fraud Detection using Machine Learning",
            "Disease Diagnosis Classification for {domain}",
            "Credit Risk Assessment Model for {domain}",
            "{domain} Sentiment Analysis Engine",
            "Spam Detection System for {domain} Communications",
            "{domain} Image Classification Pipeline",
            "Customer Segmentation for {domain} Industry"
        ],
        "tools_excellent": ["Python", "TensorFlow", "PyTorch", "XGBoost", "LightGBM", "Scikit-learn", "SHAP", "Docker", "AWS SageMaker", "MLflow"],
        "tools_good": ["Python", "Scikit-learn", "XGBoost", "Pandas", "Streamlit", "Flask"],
        "tools_average": ["Python", "Scikit-learn", "Pandas", "Matplotlib"],
        "tools_poor": ["Python", "Scikit-learn"],
        "metrics": ["accuracy", "precision", "recall", "F1-score", "AUC-ROC"],
        "improvements": ["false positives reduction", "processing time reduction", "cost savings", "detection rate improvement"]
    },
    "nlp": {
        "names": [
            "RAG-based {domain} Question Answering System",
            "{domain} Document Summarization using Transformers",
            "Named Entity Recognition for {domain} Documents",
            "Text Classification Engine for {domain}",
            "{domain} Chatbot using LLMs",
            "Topic Modeling for {domain} Research Papers",
            "Semantic Search Engine for {domain}",
            "Multi-lingual Translation System for {domain}"
        ],
        "tools_excellent": ["Python", "LangChain", "GPT-4", "BERT", "Hugging Face Transformers", "ChromaDB", "Pinecone", "FastAPI", "Docker"],
        "tools_good": ["Python", "Transformers", "BERT", "Flask", "MongoDB"],
        "tools_average": ["Python", "NLTK", "spaCy", "Gensim"],
        "tools_poor": ["Python", "NLTK"],
        "metrics": ["BLEU score", "ROUGE score", "accuracy", "F1-score", "response latency"],
        "improvements": ["query response accuracy", "processing speed", "user satisfaction", "cost reduction"]
    },
    "computer_vision": {
        "names": [
            "{domain} Object Detection System",
            "Quality Control using Computer Vision for {domain}",
            "{domain} Image Segmentation Pipeline",
            "Face Recognition System for {domain}",
            "OCR Solution for {domain} Documents",
            "Autonomous Inspection System for {domain}",
            "Medical Image Analysis for {domain}",
            "Video Analytics Platform for {domain}"
        ],
        "tools_excellent": ["Python", "PyTorch", "OpenCV", "YOLO", "Detectron2", "TensorRT", "CUDA", "AWS Rekognition", "Docker"],
        "tools_good": ["Python", "TensorFlow", "OpenCV", "Keras", "Flask"],
        "tools_average": ["Python", "OpenCV", "TensorFlow", "Keras"],
        "tools_poor": ["Python", "OpenCV"],
        "metrics": ["mAP", "IoU", "accuracy", "inference time", "FPS"],
        "improvements": ["detection accuracy", "processing speed", "false positive reduction", "latency reduction"]
    },
    "timeseries": {
        "names": [
            "{domain} Demand Forecasting System",
            "Predictive Maintenance for {domain} Equipment",
            "{domain} Stock Price Prediction Model",
            "Energy Consumption Forecasting for {domain}",
            "Anomaly Detection in {domain} Time Series",
            "{domain} Sales Forecasting Pipeline",
            "Traffic Prediction System for {domain}",
            "Weather Impact Analysis for {domain}"
        ],
        "tools_excellent": ["Python", "PyTorch", "LSTM", "Transformer", "Prophet", "Apache Spark", "Airflow", "TimescaleDB"],
        "tools_good": ["Python", "Prophet", "ARIMA", "Pandas", "Tableau"],
        "tools_average": ["Python", "Prophet", "Pandas", "Matplotlib"],
        "tools_poor": ["Python", "Pandas", "Excel"],
        "metrics": ["MAPE", "RMSE", "MAE", "R-squared"],
        "improvements": ["forecast accuracy", "inventory costs", "planning efficiency", "resource optimization"]
    },
    "recommendation": {
        "names": [
            "{domain} Personalized Recommendation Engine",
            "Collaborative Filtering System for {domain}",
            "{domain} Content-based Recommender",
            "Hybrid Recommendation Platform for {domain}",
            "Real-time Product Recommendations for {domain}",
            "{domain} User Preference Learning System",
            "Next-item Prediction for {domain}",
            "Context-aware Recommendations for {domain}"
        ],
        "tools_excellent": ["Python", "TensorFlow Recommenders", "Apache Spark", "Redis", "Elasticsearch", "Kubernetes", "A/B Testing"],
        "tools_good": ["Python", "Scikit-learn", "Surprise", "Flask", "PostgreSQL"],
        "tools_average": ["Python", "Scikit-learn", "Pandas", "Streamlit"],
        "tools_poor": ["Python", "Pandas"],
        "metrics": ["CTR", "conversion rate", "NDCG", "precision@k", "recall@k"],
        "improvements": ["engagement rate", "conversion rate", "user retention", "revenue increase"]
    },
    "deep_learning": {
        "names": [
            "Generative AI Model for {domain}",
            "{domain} Neural Network Architecture",
            "Reinforcement Learning Agent for {domain}",
            "Graph Neural Network for {domain} Analysis",
            "{domain} Attention-based Model",
            "Multi-task Learning System for {domain}",
            "Transfer Learning Pipeline for {domain}",
            "AutoML System for {domain}"
        ],
        "tools_excellent": ["Python", "PyTorch", "TensorFlow", "Weights & Biases", "Ray", "Optuna", "NVIDIA CUDA", "Docker", "Kubernetes"],
        "tools_good": ["Python", "PyTorch", "Keras", "MLflow", "Colab"],
        "tools_average": ["Python", "TensorFlow", "Keras", "Google Colab"],
        "tools_poor": ["Python", "Keras"],
        "metrics": ["loss", "accuracy", "training time", "inference latency", "model size"],
        "improvements": ["model performance", "training efficiency", "inference speed", "resource utilization"]
    }
}


def generate_ds_project(tier: str, domain: str = None) -> dict:
    """Generate a unique DS project with randomized components."""
    if domain is None:
        domain = random.choice(DS_PROJECT_DOMAINS)
    
    project_type = random.choice(list(DS_PROJECT_TYPES.keys()))
    template = DS_PROJECT_TYPES[project_type]
    
    name = random.choice(template["names"]).format(domain=domain)
    
    if tier == "excellent":
        tools = random.sample(template["tools_excellent"], min(6, len(template["tools_excellent"])))
        metric_val = random.uniform(92, 99)
        scale = random.choice(["1M+", "5M+", "10M+", "50M+", "100M+"])
        improvement = random.randint(25, 60)
    elif tier == "good":
        tools = random.sample(template["tools_good"], min(5, len(template["tools_good"])))
        metric_val = random.uniform(85, 94)
        scale = random.choice(["100K+", "500K+", "1M+"])
        improvement = random.randint(15, 35)
    elif tier == "average":
        tools = random.sample(template["tools_average"], min(4, len(template["tools_average"])))
        metric_val = random.uniform(75, 88)
        scale = random.choice(["10K+", "50K+"])
        improvement = random.randint(10, 25)
    else:  # poor
        tools = random.sample(template["tools_poor"], min(2, len(template["tools_poor"])))
        metric_val = random.uniform(65, 80)
        scale = ""
        improvement = random.randint(5, 15)
    
    metric = random.choice(template["metrics"])
    improve_type = random.choice(template["improvements"])
    
    # Generate description based on tier
    if tier == "excellent":
        desc = (f"Designed and deployed production-grade {project_type} system processing {scale} records "
                f"with {metric_val:.1f}% {metric}. Implemented state-of-the-art architectures including "
                f"ensemble methods and attention mechanisms, achieving {improvement}% {improve_type} "
                f"compared to baseline. Deployed on cloud infrastructure with auto-scaling and monitoring.")
    elif tier == "good":
        desc = (f"Built end-to-end {project_type} pipeline achieving {metric_val:.1f}% {metric} on "
                f"production dataset of {scale} samples. Performed extensive feature engineering and "
                f"hyperparameter tuning, resulting in {improvement}% {improve_type}. Created API endpoint "
                f"for model serving and integrated with existing systems.")
    elif tier == "average":
        desc = (f"Developed {project_type} model using standard ML techniques achieving {metric_val:.1f}% {metric}. "
                f"Performed data preprocessing, feature selection, and model comparison. "
                f"Achieved {improvement}% improvement over simple baseline model.")
    else:
        desc = (f"Implemented basic {project_type} using Python libraries. "
                f"Applied standard algorithms and achieved {metric_val:.1f}% {metric} on test data.")
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = ["2024", "2025", "2026"]
    date = f"{random.choice(months)} {random.choice(years)}"
    
    return {"name": name, "date": date, "tools": ", ".join(tools), "description": desc}


# ============================================================================
# EXPERIENCE TEMPLATES
# ============================================================================

DS_COMPANIES_TIER1 = ["Google", "Microsoft", "Amazon", "Meta", "Apple", "Netflix", "Uber", "Airbnb", "LinkedIn", "Twitter"]
DS_COMPANIES_TIER2 = ["Flipkart", "Razorpay", "Swiggy", "Zomato", "PhonePe", "Paytm", "Ola", "Myntra", "CRED", "Groww"]
DS_COMPANIES_TIER3 = ["Infosys", "TCS", "Wipro", "HCL", "Tech Mahindra", "Cognizant", "Capgemini", "Accenture"]
DS_COMPANIES_TIER4 = ["Local Analytics Firm", "Startup XYZ", "Freelance", "University Research Lab"]

DS_TITLES = {
    "excellent": ["Senior Data Scientist", "Lead Data Scientist", "Staff ML Engineer", "Principal Data Scientist"],
    "good": ["Data Scientist", "ML Engineer", "Applied Scientist", "Data Scientist II"],
    "average": ["Junior Data Scientist", "Data Analyst", "ML Intern", "Associate Data Scientist"],
    "poor": ["Data Science Intern", "Research Intern"]
}

DS_EXPERIENCE_BULLETS = {
    "excellent": [
        "Led team of {n} data scientists developing ML models for {product} serving {scale} users, improving {metric} by {pct}%",
        "Architected end-to-end MLOps pipeline using {tools}, reducing model deployment time from {old} to {new}",
        "Published {n} papers at {conference} on {topic}",
        "Designed and deployed real-time inference system handling {qps} queries per second with {latency}ms latency",
        "Mentored {n} junior engineers and led hiring process conducting {interviews}+ technical interviews",
        "Collaborated with cross-functional teams including product, engineering, and business to define ML strategy",
        "Reduced infrastructure costs by {pct}% through model optimization and efficient resource allocation",
        "Built feature store serving {n}+ features to {teams} ML teams across the organization"
    ],
    "good": [
        "Developed {model_type} models for {use_case} achieving {metric_val}% {metric}",
        "Built automated data pipeline processing {volume} records daily using {tools}",
        "Created dashboards and reports for business stakeholders resulting in {outcome}",
        "Implemented A/B testing framework enabling {n}+ experiments quarterly",
        "Collaborated with engineering team to deploy models with {uptime}% uptime",
        "Performed feature engineering improving model performance by {pct}%"
    ],
    "average": [
        "Analyzed {domain} data and created weekly performance reports for management",
        "Built basic dashboards using {tool} for visualizing key business metrics",
        "Assisted senior analysts in data cleaning and preprocessing tasks",
        "Developed simple ML models for {use_case} using {tools}"
    ]
}


def generate_experience_bullet(tier: str) -> str:
    """Generate a random experience bullet point."""
    templates = DS_EXPERIENCE_BULLETS.get(tier, DS_EXPERIENCE_BULLETS["average"])
    template = random.choice(templates)
    
    replacements = {
        "{n}": str(random.randint(2, 10)),
        "{scale}": random.choice(["1M+", "10M+", "50M+", "100M+", "500M+"]),
        "{metric}": random.choice(["relevance", "accuracy", "conversion", "engagement", "retention"]),
        "{pct}": str(random.randint(10, 45)),
        "{tools}": random.choice(["Vertex AI, MLflow", "SageMaker, Kubeflow", "Databricks, Airflow", "Ray, Prefect"]),
        "{old}": random.choice(["2 weeks", "1 week", "3 days"]),
        "{new}": random.choice(["4 hours", "2 hours", "30 minutes"]),
        "{conference}": random.choice(["NeurIPS", "ICML", "KDD", "AAAI", "CVPR", "ACL"]),
        "{topic}": random.choice(["recommendation systems", "NLP", "computer vision", "time series", "graph neural networks"]),
        "{qps}": random.choice(["10K", "50K", "100K", "500K"]),
        "{latency}": str(random.randint(5, 50)),
        "{interviews}": str(random.randint(20, 100)),
        "{teams}": str(random.randint(5, 20)),
        "{product}": random.choice(["search", "recommendations", "ads", "fraud detection", "pricing", "forecasting"]),
        "{model_type}": random.choice(["classification", "regression", "clustering", "NLP", "deep learning"]),
        "{use_case}": random.choice(["customer churn", "fraud detection", "demand forecasting", "sentiment analysis", "recommendations"]),
        "{metric_val}": str(random.randint(85, 96)),
        "{volume}": random.choice(["100K", "500K", "1M", "5M"]),
        "{outcome}": random.choice(["15% increase in ROI", "20% cost reduction", "data-driven decision making", "improved planning"]),
        "{uptime}": str(random.uniform(99.5, 99.99)),
        "{domain}": random.choice(["sales", "customer", "product", "marketing", "operations"]),
        "{tool}": random.choice(["Tableau", "Power BI", "Looker", "Metabase"])
    }
    
    for key, val in replacements.items():
        template = template.replace(key, str(val))
    
    return template


def generate_ds_experience(tier: str) -> list:
    """Generate experience entries based on tier."""
    experiences = []
    
    if tier == "excellent":
        companies = DS_COMPANIES_TIER1 + DS_COMPANIES_TIER2[:3]
        titles = DS_TITLES["excellent"]
        # 2 roles
        for i in range(2):
            exp = {
                "title": random.choice(titles) if i == 0 else random.choice(DS_TITLES["good"]),
                "company": random.choice(companies),
                "duration": f"{'2023' if i == 0 else '2020'} - {'Present' if i == 0 else '2023'}",
                "bullets": [generate_experience_bullet("excellent" if i == 0 else "good") for _ in range(random.randint(3, 4))]
            }
            experiences.append(exp)
    elif tier == "good":
        companies = DS_COMPANIES_TIER2 + DS_COMPANIES_TIER3[:3]
        exp = {
            "title": random.choice(DS_TITLES["good"]),
            "company": random.choice(companies),
            "duration": "2023 - Present",
            "bullets": [generate_experience_bullet("good") for _ in range(random.randint(3, 4))]
        }
        experiences.append(exp)
        # Maybe add an internship
        if random.random() > 0.4:
            intern = {
                "title": "Data Science Intern",
                "company": random.choice(DS_COMPANIES_TIER1 + DS_COMPANIES_TIER2),
                "duration": "Jan 2023 - Jun 2023",
                "bullets": [generate_experience_bullet("average") for _ in range(2)]
            }
            experiences.append(intern)
    elif tier == "average":
        if random.random() > 0.5:
            exp = {
                "title": random.choice(DS_TITLES["average"]),
                "company": random.choice(DS_COMPANIES_TIER3),
                "duration": "May 2025 - Aug 2025",
                "bullets": [generate_experience_bullet("average") for _ in range(2)]
            }
            experiences.append(exp)
    # Poor tier: no experience
    
    return experiences


# ============================================================================
# SKILLS POOLS
# ============================================================================

PROGRAMMING_LANGS = ["Python", "R", "SQL", "Scala", "Java", "C++", "Julia", "Go", "JavaScript"]
ML_FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras", "Scikit-learn", "XGBoost", "LightGBM", "CatBoost", "JAX", "MXNet"]
DL_TOOLS = ["Hugging Face Transformers", "OpenCV", "spaCy", "NLTK", "Gensim", "LangChain", "LlamaIndex"]
CLOUD_TOOLS = ["AWS (SageMaker, EC2, S3)", "GCP (Vertex AI, BigQuery)", "Azure ML", "Databricks", "Snowflake"]
BIG_DATA = ["Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Flink", "Hive", "Presto"]
MLOPS = ["Docker", "Kubernetes", "MLflow", "Kubeflow", "Weights & Biases", "DVC", "Git", "CI/CD"]
DATABASES = ["PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "Pinecone", "ChromaDB", "Weaviate"]
VIZ_TOOLS = ["Tableau", "Power BI", "Matplotlib", "Seaborn", "Plotly", "Looker", "D3.js"]
SPECIALIZED = ["NLP", "Computer Vision", "Reinforcement Learning", "LLMs", "RAG", "Time Series", "Recommendation Systems", "Graph ML"]
SOFT_SKILLS = ["Technical Leadership", "Cross-functional Collaboration", "Mentoring", "Problem Solving", "Communication", "Stakeholder Management", "Agile/Scrum"]


def generate_ds_skills(tier: str) -> dict:
    """Generate skills based on tier."""
    if tier == "excellent":
        return {
            "Programming Languages": random.sample(PROGRAMMING_LANGS, 4),
            "ML/DL Frameworks": random.sample(ML_FRAMEWORKS, 5),
            "Cloud & Big Data": random.sample(CLOUD_TOOLS + BIG_DATA, 5),
            "MLOps & Tools": random.sample(MLOPS, 5),
            "Specialized Areas": random.sample(SPECIALIZED, 4),
            "Soft Skills": random.sample(SOFT_SKILLS, 3)
        }
    elif tier == "good":
        return {
            "Programming Languages": random.sample(PROGRAMMING_LANGS[:5], 3),
            "ML Frameworks": random.sample(ML_FRAMEWORKS[:6], 4),
            "Tools & Technologies": random.sample(VIZ_TOOLS + DATABASES[:3], 4),
            "Cloud & Deployment": random.sample(CLOUD_TOOLS[:3] + MLOPS[:3], 4),
            "Specialized Areas": random.sample(SPECIALIZED, 3)
        }
    elif tier == "average":
        return {
            "Programming Languages": ["Python", "SQL"],
            "Libraries": random.sample(["Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"], 4),
            "Tools": random.sample(["Jupyter Notebook", "Git", "Tableau", "Excel", "Google Colab"], 3),
            "Areas of Interest": random.sample(SPECIALIZED[:4], 2)
        }
    else:  # poor
        return {
            "Programming": ["Python (Basic)", "SQL (Beginner)"],
            "Tools": ["Excel", "Jupyter Notebook"],
            "Learning": random.sample(["Machine Learning", "Data Science", "Statistics"], 2)
        }


# ============================================================================
# CERTIFICATIONS & ACHIEVEMENTS
# ============================================================================

CERTIFICATIONS_POOL = [
    "AWS Certified Machine Learning - Specialty",
    "Google Professional Machine Learning Engineer", 
    "TensorFlow Developer Certificate - Google",
    "Deep Learning Specialization - Coursera (Andrew Ng)",
    "Machine Learning Specialization - Stanford Online",
    "IBM Data Science Professional Certificate",
    "Azure Data Scientist Associate",
    "Databricks Certified ML Professional",
    "Python for Data Science - Coursera",
    "SQL for Data Science - Coursera",
    "Natural Language Processing Specialization - Coursera",
    "Computer Vision Nanodegree - Udacity",
    "MLOps Specialization - DeepLearning.AI",
    "Data Engineering on Google Cloud - Coursera",
    "Machine Learning with Python - IBM",
    "Applied AI with DeepLearning - Coursera",
    "Generative AI Fundamentals - Google Cloud",
    "LangChain for LLM Application Development - DeepLearning.AI"
]

ACHIEVEMENTS_POOL = [
    "Kaggle Competition Winner - {competition} (Top {rank}% of {participants}+ participants)",
    "Published paper at {conference} on {topic}",
    "Google Summer of Code {year} - Contributed to {project}",
    "AIR {rank} in GATE CS {year} ({percentile} percentile)",
    "Winner, {hackathon} Hackathon ({year})",
    "Open source contributor to {project} ({stars}+ GitHub stars)",
    "Speaker at {conference} on {topic}",
    "Research Intern at {lab} working on {topic}"
]


def generate_certifications(tier: str) -> list:
    """Generate certifications based on tier."""
    if tier == "excellent":
        return random.sample(CERTIFICATIONS_POOL[:8], random.randint(4, 5))
    elif tier == "good":
        return random.sample(CERTIFICATIONS_POOL[3:14], random.randint(3, 4))
    elif tier == "average":
        return random.sample(CERTIFICATIONS_POOL[8:], random.randint(1, 2))
    return []


def generate_achievements(tier: str) -> list:
    """Generate achievements based on tier."""
    achievements = []
    
    if tier == "excellent":
        n = random.randint(2, 3)
    elif tier == "good":
        n = random.randint(1, 2)
    else:
        return []
    
    templates = random.sample(ACHIEVEMENTS_POOL, n)
    
    for template in templates:
        ach = template.format(
            competition=random.choice(["Tabular Playground", "Home Credit Default", "Otto Product Classification", "IEEE Fraud Detection"]),
            rank=random.randint(1, 5),
            participants=random.choice(["2000", "3000", "5000", "8000"]),
            conference=random.choice(["NeurIPS", "ICML", "KDD", "PyCon India", "ODSC", "Data Science Congress"]),
            topic=random.choice(["deep learning optimization", "NLP transformers", "recommendation systems", "MLOps best practices"]),
            year=random.choice(["2023", "2024", "2025"]),
            project=random.choice(["TensorFlow", "PyTorch", "scikit-learn", "Hugging Face", "LangChain"]),
            percentile=f"{random.uniform(99.0, 99.9):.1f}",
            hackathon=random.choice(["Smart India", "Google Code Jam", "Intel AI", "Microsoft Imagine Cup"]),
            lab=random.choice(["Microsoft Research", "Google AI", "IIIT-H ML Lab", "IIT Bombay AI Lab"]),
            stars=random.choice(["500", "1000", "2000", "5000"])
        )
        achievements.append(ach)
    
    return achievements


# ============================================================================
# EDUCATION GENERATION
# ============================================================================

def generate_education(tier: str) -> list:
    """Generate education based on tier."""
    education = []
    
    if tier == "excellent":
        # M.Tech + B.Tech
        college1 = random.choice(COLLEGES_TIER1)
        college2 = random.choice(COLLEGES_TIER1 + COLLEGES_TIER2)
        education.append({
            "institution": college1,
            "degree": random.choice(["M.Tech in Computer Science (Machine Learning)", "M.Tech in AI", "MS in Data Science"]),
            "year": "2019 - 2021",
            "score": f"CGPA: {random.uniform(8.8, 9.8):.2f}/10"
        })
        education.append({
            "institution": college2,
            "degree": "B.Tech in Computer Science and Engineering",
            "year": "2015 - 2019",
            "score": f"CGPA: {random.uniform(8.5, 9.5):.2f}/10"
        })
    elif tier == "good":
        college = random.choice(COLLEGES_TIER2 + COLLEGES_TIER3)
        education.append({
            "institution": college,
            "degree": random.choice(["B.Tech in Computer Science", "B.Tech in AI & Data Science", "B.Tech in Information Technology"]),
            "year": "2020 - 2024",
            "score": f"CGPA: {random.uniform(8.0, 9.2):.2f}/10"
        })
    elif tier == "average":
        college = random.choice(COLLEGES_TIER3 + COLLEGES_TIER4)
        education.append({
            "institution": college,
            "degree": random.choice(["B.Tech in IT", "B.Tech in CSE", "B.Sc in Computer Science"]),
            "year": "2021 - 2025",
            "score": f"CGPA: {random.uniform(7.0, 8.2):.2f}/10"
        })
        # Add 12th
        education.append({
            "institution": random.choice(SCHOOLS),
            "degree": f"Class XII - {random.choice(BOARDS)}",
            "year": "2021",
            "score": f"Percentage: {random.uniform(75, 90):.1f}%"
        })
    else:  # poor
        college = random.choice(COLLEGES_TIER4)
        education.append({
            "institution": college,
            "degree": "B.Sc in Computer Science",
            "year": "2022 - 2025",
            "score": f"Percentage: {random.uniform(55, 70):.1f}%"
        })
    
    return education


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_summary(tier: str, name: str) -> str:
    """Generate professional summary based on tier."""
    specializations = random.sample(SPECIALIZED, 2)
    
    if tier == "excellent":
        years = random.randint(5, 8)
        templates = [
            f"Results-driven Data Scientist with {years}+ years of experience in machine learning, deep learning, and AI at scale. Led teams at top tech companies deploying ML systems serving millions of users. Expertise in {specializations[0]} and {specializations[1]}. Published researcher with papers at top ML conferences. Passionate about building impactful AI solutions.",
            f"Senior ML Engineer with {years}+ years building production ML systems at scale. Deep expertise in {specializations[0]} and {specializations[1]} with proven track record of delivering business impact. Led cross-functional teams and mentored junior engineers. Active contributor to open-source ML projects.",
            f"Principal Data Scientist specializing in {specializations[0]} and {specializations[1]} with {years}+ years of industry experience. Architected ML platforms serving 100M+ users. Published {random.randint(3, 8)} papers at top venues. Passionate about translating cutting-edge research into production systems."
        ]
    elif tier == "good":
        years = random.randint(2, 4)
        templates = [
            f"Motivated Data Scientist with {years}+ years of experience in machine learning and data analysis. Strong foundation in Python, statistics, and ML algorithms with hands-on experience in {specializations[0]}. Proven track record of delivering business impact through data-driven solutions. Eager to tackle complex problems.",
            f"ML Engineer with {years} years of experience building end-to-end machine learning pipelines. Skilled in {specializations[0]} and {specializations[1]}. Strong communicator who enjoys collaborating with cross-functional teams to deliver impactful solutions.",
            f"Data Scientist passionate about applying ML to solve real-world problems. {years}+ years of experience in {specializations[0]} with strong Python and SQL skills. Enthusiastic about continuous learning and staying updated with latest developments in AI."
        ]
    elif tier == "average":
        templates = [
            f"Final year B.Tech student with foundational knowledge in data science and machine learning. Familiar with Python, SQL, and basic ML algorithms. Completed internship in data analytics. Looking for entry-level opportunities in data science.",
            f"Aspiring Data Scientist with academic background in computer science. Knowledge of Python, machine learning basics, and data analysis. Completed coursework projects in {specializations[0]}. Eager to apply classroom learning to real-world problems.",
            f"Recent graduate with interest in data science and analytics. Basic proficiency in Python, SQL, and visualization tools. Completed online certifications in machine learning. Seeking opportunities to learn and grow in the field."
        ]
    else:
        templates = [
            "Recent graduate interested in data science and machine learning. Basic knowledge of Python and SQL. Completed online courses and personal projects. Looking for entry-level opportunities to learn and grow.",
            "Fresher with interest in data analysis and machine learning. Learning Python and SQL through online courses. Seeking internship opportunities to gain practical experience.",
            "Entry-level candidate interested in data science field. Familiar with basic Python and Excel. Completed introductory machine learning course. Eager to start career in analytics."
        ]
    
    return random.choice(templates)


# ============================================================================
# MAIN RESUME GENERATORS
# ============================================================================

def generate_unique_name(used_names: set) -> str:
    """Generate a unique name not in the used set."""
    while True:
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        if name not in used_names:
            used_names.add(name)
            return name


def generate_ds_resume(index: int, used_names: set) -> dict:
    """Generate a unique Data Science resume."""
    name = generate_unique_name(used_names)
    
    # Determine tier
    if index < 10:
        tier = "excellent"
    elif index < 20:
        tier = "good"
    elif index < 25:
        tier = "average"
    else:
        tier = "poor"
    
    # Generate unique projects
    num_projects = 3 if tier in ["excellent", "good"] else (2 if tier == "average" else 1)
    domains = random.sample(DS_PROJECT_DOMAINS, num_projects)
    projects = [generate_ds_project(tier, domain) for domain in domains]
    
    return {
        "name": name, 
        "tier": tier, 
        "role": "Data Scientist",
        "location": random.choice(CITIES),
        "phone": f"+91 {random.randint(6000000000, 9999999999)}",
        "email": f"{name.lower().replace(' ', '.')}@gmail.com",
        "linkedin": f"linkedin.com/in/{name.lower().replace(' ', '-')}",
        "github": f"github.com/{name.lower().replace(' ', '')}",
        "summary": generate_summary(tier, name),
        "education": generate_education(tier),
        "skills": generate_ds_skills(tier),
        "experience": generate_ds_experience(tier),
        "projects": projects,
        "certifications": generate_certifications(tier),
        "achievements": generate_achievements(tier)
    }


# ============================================================================
# OTHER ROLES
# ============================================================================

OTHER_ROLES = {
    "Software Engineer": {
        "skills": {
            "Languages": ["Java", "Python", "JavaScript", "TypeScript", "Go", "C++", "Rust"],
            "Frameworks": ["Spring Boot", "React", "Node.js", "Django", "FastAPI", "Express.js"],
            "Databases": ["PostgreSQL", "MongoDB", "Redis", "MySQL", "Elasticsearch"],
            "DevOps": ["Docker", "Kubernetes", "AWS", "GCP", "CI/CD", "Terraform"]
        },
        "project_templates": [
            {"name": "Microservices {domain} Platform", "tools": ["Java", "Spring Boot", "Kubernetes", "PostgreSQL", "Redis"]},
            {"name": "Real-time {domain} System", "tools": ["Node.js", "React", "WebSocket", "MongoDB"]},
            {"name": "{domain} API Gateway", "tools": ["Go", "gRPC", "Redis", "Docker"]},
            {"name": "Distributed {domain} Service", "tools": ["Python", "FastAPI", "Kafka", "PostgreSQL"]}
        ]
    },
    "Product Manager": {
        "skills": {
            "Product Skills": ["Product Strategy", "Roadmapping", "User Research", "A/B Testing", "Agile/Scrum", "PRDs"],
            "Technical Skills": ["SQL", "Data Analysis", "JIRA", "Figma", "Amplitude", "Mixpanel"],
            "Soft Skills": ["Stakeholder Management", "Cross-functional Leadership", "Communication", "Prioritization"]
        },
        "project_templates": [
            {"name": "{domain} Feature Launch", "tools": ["JIRA", "Figma", "SQL", "Amplitude"]},
            {"name": "{domain} User Experience Redesign", "tools": ["Figma", "Maze", "SQL", "Hotjar"]},
            {"name": "{domain} Product Growth Initiative", "tools": ["SQL", "A/B Testing", "Mixpanel"]}
        ]
    },
    "DevOps Engineer": {
        "skills": {
            "Cloud Platforms": ["AWS (EC2, EKS, Lambda)", "GCP (GKE, Cloud Run)", "Azure"],
            "IaC": ["Terraform", "Ansible", "CloudFormation", "Pulumi"],
            "Containers": ["Docker", "Kubernetes", "Helm", "ArgoCD", "Istio"],
            "Monitoring": ["Prometheus", "Grafana", "ELK Stack", "Datadog", "New Relic"]
        },
        "project_templates": [
            {"name": "{domain} Infrastructure Migration", "tools": ["Kubernetes", "Terraform", "ArgoCD"]},
            {"name": "{domain} CI/CD Pipeline", "tools": ["Jenkins", "GitLab CI", "Docker", "SonarQube"]},
            {"name": "{domain} Monitoring Platform", "tools": ["Prometheus", "Grafana", "ELK Stack"]}
        ]
    },
    "Frontend Developer": {
        "skills": {
            "Languages": ["JavaScript", "TypeScript", "HTML5", "CSS3"],
            "Frameworks": ["React", "Next.js", "Vue.js", "Angular", "TailwindCSS"],
            "Tools": ["Webpack", "Vite", "Jest", "Cypress", "Storybook"],
            "Other": ["GraphQL", "REST APIs", "Accessibility", "Performance"]
        },
        "project_templates": [
            {"name": "{domain} Progressive Web App", "tools": ["React", "Next.js", "TailwindCSS", "PWA"]},
            {"name": "{domain} Design System", "tools": ["React", "TypeScript", "Storybook", "Figma"]},
            {"name": "{domain} Dashboard", "tools": ["Vue.js", "D3.js", "TypeScript"]}
        ]
    },
    "Backend Developer": {
        "skills": {
            "Languages": ["Python", "Go", "Java", "Node.js", "Rust"],
            "Frameworks": ["FastAPI", "Django", "Spring Boot", "Express", "Gin"],
            "Databases": ["PostgreSQL", "MongoDB", "Redis", "Cassandra"],
            "Other": ["REST APIs", "GraphQL", "gRPC", "Kafka", "RabbitMQ"]
        },
        "project_templates": [
            {"name": "{domain} API Service", "tools": ["Python", "FastAPI", "PostgreSQL", "Redis"]},
            {"name": "{domain} Event-Driven System", "tools": ["Go", "Kafka", "MongoDB", "Docker"]},
            {"name": "{domain} Payment Gateway", "tools": ["Java", "Spring Boot", "PostgreSQL"]}
        ]
    },
    "Data Engineer": {
        "skills": {
            "Languages": ["Python", "SQL", "Scala", "Java"],
            "Big Data": ["Apache Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Flink"],
            "Databases": ["PostgreSQL", "Snowflake", "BigQuery", "Redshift", "Delta Lake"],
            "Cloud": ["AWS (EMR, Glue)", "GCP (Dataflow)", "Databricks"]
        },
        "project_templates": [
            {"name": "{domain} Data Pipeline", "tools": ["Spark", "Airflow", "Delta Lake", "AWS"]},
            {"name": "{domain} Data Warehouse", "tools": ["dbt", "Snowflake", "Airflow"]},
            {"name": "{domain} Streaming Platform", "tools": ["Kafka", "Flink", "PostgreSQL"]}
        ]
    },
    "Mobile Developer": {
        "skills": {
            "Languages": ["Swift", "Kotlin", "Dart", "JavaScript"],
            "Frameworks": ["iOS (SwiftUI)", "Android (Jetpack)", "Flutter", "React Native"],
            "Tools": ["Xcode", "Android Studio", "Firebase", "Fastlane"],
            "Other": ["Push Notifications", "In-App Purchases", "Analytics"]
        },
        "project_templates": [
            {"name": "{domain} Mobile App", "tools": ["Flutter", "Firebase", "Bloc"]},
            {"name": "{domain} iOS Application", "tools": ["Swift", "SwiftUI", "Core Data"]},
            {"name": "{domain} Android App", "tools": ["Kotlin", "Jetpack Compose", "Room"]}
        ]
    },
    "QA Engineer": {
        "skills": {
            "Automation": ["Selenium", "Appium", "Playwright", "Cypress", "RestAssured"],
            "Languages": ["Python", "Java", "JavaScript"],
            "Tools": ["JIRA", "TestRail", "Postman", "JMeter", "k6"],
            "Testing": ["API Testing", "UI Testing", "Performance Testing", "Security Testing"]
        },
        "project_templates": [
            {"name": "{domain} Test Automation Framework", "tools": ["Python", "Selenium", "pytest"]},
            {"name": "{domain} API Test Suite", "tools": ["Postman", "RestAssured", "Newman"]},
            {"name": "{domain} Performance Testing", "tools": ["JMeter", "k6", "Grafana"]}
        ]
    },
    "Security Engineer": {
        "skills": {
            "Security": ["Penetration Testing", "Vulnerability Assessment", "SAST/DAST", "Threat Modeling"],
            "Tools": ["Burp Suite", "Nessus", "Metasploit", "OWASP ZAP", "Snyk"],
            "Cloud Security": ["AWS Security", "Azure Security", "Kubernetes Security"],
            "Languages": ["Python", "Bash", "Go"]
        },
        "project_templates": [
            {"name": "{domain} Security Automation", "tools": ["Python", "Docker", "SAST/DAST"]},
            {"name": "{domain} Zero Trust Implementation", "tools": ["AWS", "Vault", "Kubernetes"]},
            {"name": "{domain} Bug Bounty Program", "tools": ["Burp Suite", "OWASP", "Reporting"]}
        ]
    },
    "UX Designer": {
        "skills": {
            "Design Tools": ["Figma", "Sketch", "Adobe XD", "Principle", "Framer"],
            "Research": ["User Interviews", "Usability Testing", "A/B Testing", "Surveys"],
            "Technical": ["HTML/CSS", "Prototyping", "Design Systems", "Accessibility"],
            "Soft Skills": ["Visual Design", "Interaction Design", "Information Architecture"]
        },
        "project_templates": [
            {"name": "{domain} App Redesign", "tools": ["Figma", "Maze", "Hotjar"]},
            {"name": "{domain} Design System", "tools": ["Figma", "Storybook", "Zeroheight"]},
            {"name": "{domain} User Research Study", "tools": ["Interviews", "Surveys", "Affinity Mapping"]}
        ]
    }
}

OTHER_COMPANIES = [
    "Microsoft", "Google", "Amazon", "Meta", "Apple", "Netflix", "Uber", "Airbnb",
    "Flipkart", "Razorpay", "Swiggy", "Zomato", "PhonePe", "Paytm", "Ola", "Myntra", "CRED",
    "Infosys", "TCS", "Wipro", "Cognizant", "Accenture", "Deloitte", "McKinsey"
]

OTHER_DOMAINS = ["E-commerce", "Fintech", "Healthcare", "EdTech", "Social Media", "Logistics", "Travel", "Gaming"]


def generate_other_project(role: str, domain: str) -> dict:
    """Generate a project for non-DS role."""
    role_data = OTHER_ROLES[role]
    template = random.choice(role_data["project_templates"])
    
    name = template["name"].format(domain=domain)
    tools = ", ".join(random.sample(template["tools"], min(4, len(template["tools"]))))
    
    # Generate description
    metrics = random.randint(15, 50)
    scale = random.choice(["10K", "100K", "500K", "1M", "5M"])
    
    descriptions = [
        f"Designed and implemented {name.lower()} serving {scale}+ users. Improved {random.choice(['performance', 'reliability', 'user experience', 'efficiency'])} by {metrics}% through careful architecture and optimization.",
        f"Led development of {name.lower()} from concept to production. Collaborated with cross-functional teams to deliver features that increased {random.choice(['engagement', 'conversion', 'retention', 'satisfaction'])} by {metrics}%.",
        f"Built end-to-end {name.lower()} handling {scale}+ transactions. Implemented {random.choice(['caching', 'async processing', 'microservices', 'event-driven'])} architecture achieving {random.choice(['99.9%', '99.99%'])} uptime."
    ]
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    date = f"{random.choice(months)} {random.choice(['2024', '2025', '2026'])}"
    
    return {"name": name, "date": date, "tools": tools, "description": random.choice(descriptions)}


def generate_other_resume(index: int, used_names: set) -> dict:
    """Generate a unique non-DS resume."""
    name = generate_unique_name(used_names)
    roles = list(OTHER_ROLES.keys())
    role = roles[index % len(roles)]
    role_data = OTHER_ROLES[role]
    
    # Generate unique skills by sampling
    skills = {}
    for category, skill_list in role_data["skills"].items():
        skills[category] = random.sample(skill_list, min(len(skill_list), random.randint(3, 5)))
    
    # Generate projects
    domains = random.sample(OTHER_DOMAINS, 2)
    projects = [generate_other_project(role, domain) for domain in domains]
    
    # Generate experience
    company = random.choice(OTHER_COMPANIES)
    experience = [{
        "title": role,
        "company": company,
        "duration": f"{random.choice(['2022', '2023'])} - Present",
        "bullets": [
            f"Developed and maintained {random.choice(['core features', 'production systems', 'customer-facing products'])} serving {random.choice(['100K', '1M', '5M'])}+ users",
            f"Improved {random.choice(['system performance', 'code quality', 'team velocity', 'deployment frequency'])} by {random.randint(20, 50)}% through {random.choice(['automation', 'refactoring', 'best practices', 'tooling'])}",
            f"Collaborated with {random.choice(['product', 'design', 'backend', 'frontend', 'QA'])} teams to deliver {random.randint(5, 15)}+ features",
            f"Mentored {random.randint(2, 5)} junior {role.lower()}s on {random.choice(['best practices', 'code reviews', 'architecture', 'testing'])}"
        ]
    }]
    
    return {
        "name": name,
        "tier": "good",
        "role": role,
        "location": random.choice(CITIES),
        "phone": f"+91 {random.randint(6000000000, 9999999999)}",
        "email": f"{name.lower().replace(' ', '.')}@gmail.com",
        "linkedin": f"linkedin.com/in/{name.lower().replace(' ', '-')}",
        "github": f"github.com/{name.lower().replace(' ', '')}",
        "summary": f"Experienced {role} with {random.randint(2, 5)}+ years of expertise in building scalable solutions. Strong technical foundation with proven track record of delivering impactful projects. Passionate about {random.choice(['clean code', 'user experience', 'system design', 'continuous improvement'])}.",
        "education": [{
            "institution": random.choice(COLLEGES_TIER2 + COLLEGES_TIER3),
            "degree": "B.Tech in Computer Science",
            "year": f"{random.randint(2016, 2020)} - {random.randint(2020, 2024)}",
            "score": f"CGPA: {random.uniform(7.5, 9.0):.2f}/10"
        }],
        "skills": skills,
        "experience": experience,
        "projects": projects,
        "certifications": [],
        "achievements": []
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Generating 50 UNIQUE test resumes as PDFs...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    (OUTPUT_DIR / "data_science").mkdir(exist_ok=True)
    (OUTPUT_DIR / "other_roles").mkdir(exist_ok=True)
    
    used_names = set()
    
    print("\n--- Generating 30 Data Science Resumes ---")
    for i in range(30):
        data = generate_ds_resume(i, used_names)
        filename = f"ds_{i+1:02d}_{data['tier']}_{data['name'].replace(' ', '_')}.pdf"
        filepath = OUTPUT_DIR / "data_science" / filename
        create_resume_pdf(data, filepath)
        print(f"  [{i+1:02d}/30] {data['tier'].upper():10s} - {data['name']}")
    
    print("\n--- Generating 20 Other Role Resumes ---")
    for i in range(20):
        data = generate_other_resume(i, used_names)
        role_short = data['role'].lower().replace(' ', '_')
        filename = f"other_{i+1:02d}_{role_short}_{data['name'].replace(' ', '_')}.pdf"
        filepath = OUTPUT_DIR / "other_roles" / filename
        create_resume_pdf(data, filepath)
        print(f"  [{i+1:02d}/20] {data['role']:20s} - {data['name']}")
    
    print(f"\n✅ Generated 50 UNIQUE PDF resumes in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
