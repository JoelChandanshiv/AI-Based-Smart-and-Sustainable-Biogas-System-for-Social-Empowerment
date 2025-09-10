# 🌱 AI-Based-Smart-and-Sustainable-Biogas-System-for-Social-Empowerment
🚀 Smart Biogas Production & Monitoring System powered by AI + IoT + GenAI

This project showcases a complete end-to-end solution for biogas production optimization:

- AI-driven biogas quality prediction using sensors

- Real-time dashboards and mobile apps

- A GenAI chatbot and AI agent for farmer/plant operator support

![System Design](/ECONOVA.jpg)

---

## 📌 Problem Statement

Conventional biogas plants face two major challenges:

1️⃣ Unpredictable Biogas Quality – Lack of real-time monitoring reduces efficiency.

2️⃣ Limited Decision Support – Farmers/operators don’t get timely insights or corrective measures.

✅ Our solution tackles both problems with AI-driven prediction, IoT automation, and GenAI decision support.

---

## 🌟 Key Features

✅ **Hybrid Biogas Quality Prediction** – LSTM + XGBoost model using sensor data (pH, temperature, methane %, CO₂ %, pressure).

✅ **IoT Integration** – Real-time sensors continuously stream data.

✅ **Mobile Application** – Farmers/operators can monitor plant health, see dashboards, and chat with a GenAI assistant.

✅ **Power BI Dashboard** – Interactive visualization of energy produced, efficiency trends, and predictions.

✅ **GenAI Chatbot** – Integrated inside mobile app (mock using Ollama/GPT4All/free LLM).

✅ **AI Agent** – Automates tasks like alerting, optimization suggestions, and report generation.

---

## 📝Summary of Prototype
EcoNova: A Smart and Sustainable Biogas System for Rural Empowerment. 

Many farmers and rural communities struggle with low biogas production and poor-quality slurry from organic waste. Existing systems are inefficient, unsafe, and lack proper monitoring. There is a need for a simple, affordable solution to improve biogas yield, enrich slurry, and reduce pathogens for better use in farming. To design and implement an efficient, affordable, and user-friendly biogas system that maximizes gas production, enriches slurry for agricultural use, reduces harmful pathogens, and incorporates IoT-based monitoring, empowering rural communities with sustainable energy and high-quality organic fertilizer solutions. EcoNova is a sustainable biogas system that efficiently processes organic waste to produce renewable energy and high-quality fertilizer. The system begins by mixing organic waste with water in a mixing tank, then uses anaerobic digestion in a digester to generate biogas. Key parameters such as temperature, pressure, gas production rate, and slurry quality are monitored in real-time through IoT monitoring to ensure optimal system performance. The biogas is purified through water scrubbers to remove harmful gases like hydrogen sulphide and carbon dioxide, while the leftover slurry undergoes UV disinfection to eliminate pathogens. Finally, the slurry is enriched with nutrients in an enrichment tank, turning it into a safe, nutrient-rich fertilizer through the compost tea extractor method, which includes ingredients like corn leaves, roots, stalks, and vermicompost. This process helps reduce the salinity of the slurry, making it usable for all types of farms. Continuous oxygen is supplied during this process to ensure optimal conditions. This integrated solution empowers rural communities by providing clean energy and enhancing agricultural productivity in a simple, affordable, and environmentally friendly way.

---
## 🎯 Problem Being Addressed  

Improper waste management and reliance on costly, polluting energy sources pose significant challenges in rural and urban regions. This project directly addresses:  

- **Waste Disposal Issues** → Reduces landfill waste by converting organic matter into useful resources.  
- **Energy Shortages** → Provides a renewable and reliable energy source for cooking, heating, and power generation.  
- **Agricultural Support** → Supplies natural fertilizer, reducing dependency on chemical fertilizers.  


--- 

### Launch Mobile App

* Open `mobile_app/` in **Android Studio** (for Kotlin) .
* Run the emulator or connect a physical device.

![Mobile Application](mobile_app/ECONOVA/Mobile%20APP%20UI.jpeg)


## 📊 Dashboard (Power BI)

![Power BI Dashboard](dashboard/powerBI.jpeg)

---

## 🤖 GenAI Chatbot & AI Agent

* **Chatbot** → Built using a free local LLM (Ollama). Integrated into the mobile app for answering farmer/operator queries.
* **AI Agent** → Uses LangChain/CrewAI to automate:

  * Alerts when methane % is too low
  * Generates daily plant performance report
  * Suggests corrective actions (e.g., “Add more organic waste, pH too low”)

---

🔥 With this project, we are not just **producing renewable energy**, but also **building the future of sustainable waste management using AI + IoT + GenAI**.
