import spacy
import os

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from __future__ import division, print_function
import os
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import spacy
from googletrans import Translator
import io
from flask import send_file

# Define a Flask app
app = Flask(__name__)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size is 16 MB

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

# Path to the trained PyTorch model
MODEL_PATH = 'plant_disease_model.pth'

# Class labels (replace with your actual class names)
CLASS_LABELS = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# A dictionary mapping the class labels to descriptions and solutions
DESCRIPTIONS = {
  "Apple___Apple_scab": {
    "description": "Apple scab is one of the most common fungal diseases affecting apple trees, caused by the pathogen *Venturia inaequalis*. This disease primarily impacts the leaves, fruit, and shoots of the tree, causing dark, olive-green to black lesions. Over time, these lesions expand and cause premature leaf drop, reducing the tree's ability to photosynthesize effectively. Infected fruits develop dark spots and become deformed, which significantly reduces their market value. Apple scab is most prevalent during wet, rainy spring weather when the spores are dispersed by wind and rain. The disease can spread rapidly if not controlled early, leading to a substantial decline in tree health and fruit yield.",
    "solution": "Effective management of apple scab requires both preventative and curative strategies. Start by choosing resistant apple varieties, which are less susceptible to the disease. Apply fungicides early in the growing season, particularly during periods of wet weather, as the fungus thrives in moisture. Fungicides containing copper or sulfur are particularly effective. Prune the tree to improve airflow and reduce humidity around the leaves, which helps to prevent the development of spores. Additionally, remove and destroy fallen leaves and infected fruit in the fall to reduce the chances of spores overwintering. Keep the orchard clean by raking and disposing of debris and considering mulch to prevent further spread."
  },
  "Apple___Black_rot": {
    "description": "Black rot is a severe fungal disease caused by *Alternaria alternata* that affects apples, causing dark, sunken lesions on the fruit and cankers on the branches. The disease spreads rapidly through the tree, especially in warm, wet conditions. Infected apples turn black, shrivel, and rot, while cankers form on branches and spread into the tree, causing dieback. Black rot can also cause fruit drop, leading to a reduced harvest. It often occurs when the tree is stressed by excessive moisture, poor nutrition, or injury, and it is more common in older, weakened trees. The fungus that causes black rot can survive in dead tissue on the tree and in fallen fruit, making it essential to remove all infected parts.",
    "solution": "To control black rot, start by removing all infected fruit, branches, and twigs promptly to prevent the disease from spreading. Prune trees to remove any affected or dead wood, and ensure good air circulation within the tree canopy. Fungicides, such as copper-based products, can be effective when applied during the growing season, especially during periods of high humidity or rain. It is also critical to avoid overhead irrigation, as wet foliage encourages the spread of the disease. Additionally, proper fertilization and irrigation will help reduce tree stress, making the plant more resistant to disease. Remove any fallen fruit and debris from the ground as they can serve as a source of reinfection."
  },
  "Apple___Cedar_apple_rust": {
    "description": "Cedar apple rust is a fungal disease caused by *Gymnosporangium juniperi-virginianae*, which requires both apple trees and juniper trees (cedars) for its lifecycle. The disease is characterized by bright yellow-orange spots that appear on the upper surface of apple leaves, often surrounded by a dark border. Infected leaves become distorted, and over time, they may drop prematurely, weakening the tree. The fungus also produces large, galls on juniper trees, which release spores that can travel long distances in the wind. Although cedar apple rust does not typically kill apple trees outright, it significantly weakens them, reducing fruit production and overall health. The disease is most prevalent in areas with a high population of both apple and juniper trees.",
    "solution": "Control of cedar apple rust requires a multi-faceted approach. The first step is to remove or reduce the number of juniper trees (cedars) near apple orchards, as they are an essential part of the disease's life cycle. Pruning infected apple leaves and removing them from the orchard can help reduce the number of spores present. Fungicide treatments, such as those containing copper or sulfur, should be applied in early spring, before infection occurs, and continued throughout the growing season during periods of high humidity or rain. Proper spacing between trees and good air circulation can help reduce the moist conditions that favor fungal growth. Regular monitoring for early signs of infection is essential to manage this disease."
  },
  "Apple___healthy": {
    "description": "A healthy apple tree exhibits strong, vigorous growth, with vibrant green leaves and well-formed fruit. The tree should have no signs of wilting, yellowing, or premature leaf drop. Healthy apple trees produce abundant fruit with minimal pest or disease pressure. The bark should be smooth and free from cankers, and there should be no sign of dieback in branches or roots. A healthy apple tree is also well-rooted, with roots that spread far enough to supply the tree with sufficient water and nutrients. To maintain tree health, it is important to provide the right growing conditions and to regularly inspect for pests, diseases, and nutritional deficiencies.",
    "solution": "Maintaining a healthy apple tree requires proper care and management. Start by planting in well-drained soil with good fertility and sunlight, as apple trees need at least 6 hours of direct sunlight each day. Prune trees regularly to remove dead or diseased wood and to improve air circulation, which helps reduce the likelihood of fungal infections. Fertilize with a balanced fertilizer that provides the necessary nutrients for growth, especially nitrogen, phosphorus, and potassium. Water the tree consistently but avoid overwatering, as apple trees do not like wet feet. Apply mulch around the base of the tree to retain moisture and suppress weeds. Lastly, regularly inspect the tree for pests and diseases, and take immediate action to manage any issues that arise."
  },
  "Blueberry___healthy": {
    "description": "Healthy blueberry bushes are characterized by strong, vibrant foliage, and the fruits are firm and plump with a rich blue color. The leaves of healthy bushes are dark green and free from blemishes, spots, or discoloration. Healthy blueberry bushes are well-established, with a strong root system that helps them access water and nutrients efficiently. These plants thrive in acidic, well-drained soils and require a stable environment with adequate sunlight, moisture, and protection from extreme weather conditions. A healthy blueberry plant also shows no signs of pest infestations or disease, which are common threats in blueberry cultivation.",
    "solution": "To maintain healthy blueberry bushes, ensure they are planted in acidic, well-drained soil with a pH between 4.5 and 5.5. Blueberries require full sun for at least 6 hours per day, so plant them in a location that receives ample light. Water the plants regularly, especially during dry periods, but be careful not to overwater, as this can lead to root rot. Fertilize with a specialized fertilizer for acid-loving plants to provide essential nutrients. Mulch around the base of the bushes to retain moisture, suppress weeds, and maintain a consistent temperature. Prune bushes annually to remove dead or damaged wood, and keep the plant's shape open to allow for better air circulation. Regularly inspect for pests and diseases such as aphids, mites, or fungal infections, and take prompt action if needed."
  },
  "Cherry_(including_sour)___Powdery_mildew": {
    "description": "Powdery mildew, caused by the fungus *Podosphaera leucotricha*, affects the leaves, flowers, and young shoots of cherry trees. It appears as a white, powdery coating on the surface of the leaves. Infected leaves may become distorted, and the tree's growth may be stunted. The disease can also cause fruit to develop poorly or fail to mature. Powdery mildew thrives in warm, dry conditions with high humidity and can spread rapidly if not controlled.",
    "solution": "To manage powdery mildew, prune and remove infected plant parts, and dispose of them away from the garden. Apply fungicides, particularly during the early stages of the disease, when the fungus is most vulnerable. Increase air circulation around the tree by spacing it properly and pruning to avoid dense foliage. Avoid overhead irrigation, as wet foliage can encourage fungal growth. If the problem persists, consider using resistant varieties of cherry trees."
  },
  "Cherry_(including_sour)___healthy": {
    "description": "Healthy cherry trees display vibrant green leaves and produce healthy fruit without signs of disease or pest damage. The tree should have a well-developed root system, producing uniform, flavorful cherries. Cherry trees thrive in well-drained soil with a slightly acidic to neutral pH and require full sun for optimal growth. A healthy tree also has strong resistance to common diseases and pests.",
    "solution": "To keep cherry trees healthy, ensure they are planted in well-drained soil with proper irrigation. Regularly prune dead or diseased wood to maintain good airflow and encourage strong growth. Fertilize with a balanced fertilizer designed for fruit-bearing trees. Monitor for pests and diseases, and take preventive action early to avoid issues. Mulching around the base of the tree helps retain moisture and prevents weeds."
  },
  "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
    "description": "Cercospora leaf spot and gray leaf spot are fungal diseases caused by *Cercospora zeae-maydis* and *Cochliobolus heterostrophus*, respectively. They cause the leaves to develop grayish-brown lesions with dark borders. The disease can lead to premature leaf drop, reducing photosynthesis and weakening the plant. These diseases are most common in hot, humid conditions and can significantly reduce yield if not managed.",
    "solution": "To manage Cercospora and gray leaf spot, remove infected leaves and dispose of them to limit the spread of spores. Apply fungicides early in the growing season, especially during wet periods. Practice crop rotation to reduce the buildup of fungal spores in the soil. Plant resistant maize varieties, and ensure that plants are spaced adequately to reduce humidity around the plants."
  },
  "Corn_(maize)___Common_rust_": {
    "description": "Common rust, caused by *Puccinia sorghi*, is a fungal disease that affects maize plants. It produces orange-red pustules on the leaves, which can cause premature leaf senescence and reduce photosynthesis. If severe, it can lead to stunted growth and yield loss. The disease spreads rapidly in hot, humid conditions and can significantly damage crops if not controlled.",
    "solution": "To control common rust, apply fungicides during early plant growth, particularly if rust symptoms are observed. Prune away infected leaves to reduce the spread of the disease. Rotate crops to reduce rust spores in the soil, and plant rust-resistant maize varieties. Avoid overhead irrigation and ensure proper plant spacing for adequate airflow."
  },
  "Corn_(maize)___Northern_Leaf_Blight": {
    "description": "Northern leaf blight, caused by *Exserohilum turcicum*, is a fungal disease that causes long, grayish-green lesions on maize leaves. The lesions eventually turn brown and cause the leaves to die. The disease spreads rapidly during wet weather and can significantly reduce yield by affecting the plant’s ability to photosynthesize. Severe infections can lead to stalk rot and increased vulnerability to other diseases.",
    "solution": "To manage northern leaf blight, remove and destroy infected plant material to reduce the source of fungal spores. Apply fungicides during early plant growth to protect the leaves. Use resistant maize varieties and practice crop rotation to limit the buildup of the fungus in the soil. Ensure proper spacing between plants for good airflow and avoid excessive moisture."
  },
  "Corn_(maize)___healthy": {
    "description": "Healthy corn plants exhibit lush, green leaves and robust growth. The plant is free from disease and pest damage, with no signs of stunted growth or nutrient deficiencies. Healthy maize plants should have well-developed roots and produce high yields of corn with well-formed ears.",
    "solution": "To maintain healthy corn plants, ensure they are planted in well-drained, fertile soil with a slightly acidic to neutral pH. Regularly water the plants to prevent drought stress, but avoid overwatering. Apply balanced fertilizers to support strong growth, and monitor for pests or diseases. Proper spacing between plants ensures adequate sunlight and airflow."
  },
  "Grape___Black_rot": {
    "description": "Black rot, caused by the fungus *Guignardia bidwellii*, affects grapevines and causes dark, sunken lesions on leaves, stems, and fruit. Infected fruit may shrivel and become mummified, and the disease can cause significant yield loss. Black rot thrives in humid conditions and spreads rapidly through the vineyard, often during rainy periods.",
    "solution": "To control black rot, prune and remove infected plant parts, including mummified fruit, and dispose of them away from the vineyard. Apply fungicides early in the season, especially during rainy weather, to protect leaves and fruit. Use resistant grape varieties where possible, and practice crop rotation to reduce the risk of reinfection. Ensure proper vine spacing and airflow to reduce humidity."
  },
  "Grape___Esca_(Black_Measles)": {
    "description": "Esca, or black measles, is a fungal disease that causes discoloration and necrosis in grapevine wood and leaves. The disease progresses slowly, but it leads to dieback in infected vines and can reduce fruit production. Symptoms include dark streaks in the wood and the death of leaves, often leading to a condition known as 'vine decline'. Esca is transmitted through pruning tools and other equipment.",
    "solution": "To manage Esca, prune away infected wood and dispose of it properly. Sanitize pruning tools regularly to prevent the spread of the disease. Avoid over-irrigation, as excessive moisture can exacerbate the disease. Ensure good vine management, with proper spacing, to promote healthy growth and minimize the stress on plants. Use resistant grape varieties if available."
  },
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
    "description": "Leaf blight, caused by *Isariopsis viticola*, is a fungal disease that causes dark, angular lesions on the leaves of grapevines. The lesions may expand to cover a large portion of the leaf, leading to premature leaf drop. This reduces the plant's ability to photosynthesize, stunting vine growth and reducing fruit yield. Leaf blight is more common in areas with high humidity and wet conditions.",
    "solution": "To manage leaf blight, prune and remove infected leaves and debris, and dispose of them away from the vineyard. Apply fungicides during the growing season to protect new growth. Space plants adequately to improve airflow and reduce humidity around the leaves. Practice good vineyard sanitation to reduce fungal spores in the environment."
  },
  "Grape___healthy": {
    "description": "Healthy grapevines have strong, green leaves and produce healthy clusters of grapes. The vine should have a robust root system and be free from disease, pests, and nutrient deficiencies. Healthy vines are well-maintained, with proper spacing to allow for optimal sunlight and airflow, ensuring a productive harvest.",
    "solution": "To maintain healthy grapevines, ensure they are planted in well-drained soil with proper irrigation. Fertilize with a balanced fertilizer to support vine growth. Regular pruning helps maintain good airflow and reduces the risk of diseases. Monitor for pests and diseases, and take prompt action if necessary. Ensure that the vines receive adequate sunlight and are spaced to allow proper air circulation."
  },
  "Orange___Haunglongbing_(Citrus_greening)": {
    "description": "Citrus greening, also known as Huanglongbing (HLB), is a bacterial disease that affects orange trees. Symptoms include yellowing leaves, misshapen fruit, and stunted growth. The bacteria spread through infected psyllids (insects), causing the tree to decline and often leading to death. The fruit may become bitter and hard, making it unsuitable for consumption.",
    "solution": "To manage citrus greening, remove and destroy infected trees to prevent the spread of the bacteria. Control psyllids with insecticides or biological control methods. Avoid planting citrus in areas prone to the disease and choose resistant varieties when possible. Regularly monitor for signs of infection, and prune infected limbs to reduce the spread of the disease."
  },
  "Peach___Bacterial_spot": {
    "description": "Bacterial spot is caused by *Xanthomonas campestris*, a bacterium that infects peach trees, leading to small, water-soaked lesions on the leaves, twigs, and fruit. Over time, these lesions turn brown or black with yellow halos around them. The disease can reduce fruit quality and yield by causing premature fruit drop, deforming fruit, and weakening the tree. In severe cases, it can cause the leaves to die, which affects the tree’s ability to photosynthesize.",
    "solution": "To manage bacterial spot, prune and remove any infected parts of the tree, especially during the dormant season. Disinfect pruning tools regularly to prevent spreading the bacteria. Apply copper-based bactericides during the growing season, particularly before and after rainstorms, to control bacterial growth. Avoid overhead irrigation and ensure good airflow around the tree to reduce humidity and prevent the spread of bacteria. In areas with frequent outbreaks, consider planting resistant peach varieties."
  },
  "Peach___healthy": {
    "description": "Healthy peach trees exhibit lush, green leaves, abundant blossoms, and high-quality fruit. The tree should be free from pests, diseases, and nutrient deficiencies, with good fruit set. Healthy peach trees have a robust root system, adequate spacing for proper airflow, and optimal exposure to sunlight. Such trees are also able to resist environmental stress and maintain strong growth.",
    "solution": "To maintain a healthy peach tree, plant it in well-drained soil with sufficient sunlight and proper irrigation. Prune the tree annually to remove dead or diseased wood, ensuring adequate air circulation and sunlight penetration. Use balanced fertilizers to support growth, and monitor the tree for early signs of pests or diseases. Mulch around the base to conserve moisture, prevent weeds, and regulate soil temperature."
  },
  "Pepper,_bell___Bacterial_spot": {
    "description": "Bacterial spot in bell peppers is caused by *Xanthomonas campestris* and results in the formation of dark, water-soaked lesions on the leaves, stems, and fruit. These lesions often have a yellow margin. The disease can cause premature leaf drop, reduce fruit quality, and weaken the plant’s overall health. Infected peppers often show stunted growth, and the fruit may develop soft, sunken spots.",
    "solution": "To manage bacterial spot, remove and destroy infected plant material. Avoid overhead irrigation and water the plants at the base to keep foliage dry. Apply copper-based bactericides, particularly during humid and rainy weather, to protect plants from further infection. Avoid working with wet plants, as this can spread the bacteria. Rotate crops to reduce the risk of reinfection, and plant resistant pepper varieties when possible."
  },
  "Pepper,_bell___healthy": {
    "description": "Healthy bell peppers are characterized by vibrant green leaves, strong stems, and large, glossy fruit that ripen to their full color. The plant should be free from disease, pests, and nutrient deficiencies. Healthy peppers grow vigorously and produce a high-quality yield, thriving in sunny conditions with proper spacing and airflow.",
    "solution": "To maintain healthy bell peppers, plant them in well-drained soil with plenty of organic matter. Regularly water the plants at the base to avoid wetting the foliage, and ensure they receive at least six hours of direct sunlight per day. Fertilize the plants with a balanced nutrient mix, and regularly inspect for pests such as aphids or aphid-transmitted viruses. Ensure proper spacing to avoid overcrowding, which can limit airflow and increase disease pressure."
  },
  "Potato___Early_blight": {
    "description": "Early blight is caused by the fungus *Alternaria solani* and manifests as small, dark spots with concentric rings on the lower leaves of potato plants. These lesions grow larger over time and cause the leaves to yellow and die, reducing the plant's ability to photosynthesize. The disease can spread rapidly under wet conditions and may lead to significant yield loss if not controlled.",
    "solution": "To control early blight, remove and destroy infected plant debris, especially at the end of the growing season. Apply fungicides at the first signs of the disease and continue applications as needed. Practice crop rotation to prevent the buildup of fungal spores in the soil, and use resistant potato varieties. Ensure proper spacing between plants to improve airflow and reduce humidity around the foliage, which helps to prevent fungal infections."
  },
  "Potato___Late_blight": {
    "description": "Late blight, caused by *Phytophthora infestans*, is one of the most devastating diseases affecting potatoes. It results in dark, water-soaked lesions on leaves, stems, and tubers. Infected plants often show wilting, and the disease can rapidly destroy the plant. Late blight thrives in cool, wet conditions and can spread quickly, leading to massive crop loss if not controlled.",
    "solution": "To manage late blight, apply fungicides at the first sign of the disease and continue treatments throughout the growing season, particularly during wet periods. Remove and destroy infected plant material to limit the spread of spores. Use resistant potato varieties, and practice crop rotation to avoid soil-borne pathogens. Ensure proper spacing between plants to promote good airflow, and avoid overhead irrigation."
  },
  "Potato___healthy": {
    "description": "Healthy potato plants have strong, green foliage, and the tubers should be growing vigorously without signs of disease or nutrient deficiencies. The plants should produce ample flowers, indicating good reproduction, and the leaves should not show symptoms of yellowing or wilting. Healthy potatoes are resilient against pests and diseases, thriving in well-drained, fertile soil.",
    "solution": "To maintain healthy potatoes, ensure they are planted in loose, well-drained soil with adequate organic matter. Regularly water the plants to maintain consistent moisture, especially during dry spells. Apply balanced fertilizers to encourage strong growth, and rotate crops to prevent soil-borne diseases. Mulch around the base of the plants to retain moisture, suppress weeds, and regulate soil temperature."
  },
  "Raspberry___healthy": {
    "description": "Healthy raspberry plants are characterized by lush, green leaves and abundant fruit production. The canes should be strong and free from diseases such as rust or mildew. The plants should show no signs of wilting, discoloration, or pest infestation. Healthy raspberries produce high-quality fruit that is sweet and firm, with no deformities.",
    "solution": "To maintain healthy raspberries, plant them in well-drained soil with plenty of organic matter. Provide full sunlight and ensure the plants have adequate space for proper airflow. Water regularly to keep the soil consistently moist but not waterlogged. Prune dead or damaged canes, and ensure the plants are well-supported to avoid disease transmission. Regularly monitor for pests, especially aphids and mites, and treat as needed."
  },
  "Soybean___healthy": {
    "description": "Healthy soybean plants have robust, green foliage, with no signs of disease or pest damage. The plants should grow uniformly and produce high-quality pods filled with healthy seeds. A healthy soybean crop also shows no signs of nutrient deficiencies, wilting, or stunted growth.",
    "solution": "To ensure healthy soybeans, plant them in well-drained, fertile soil with adequate moisture. Provide a balanced fertilizer to meet the plant’s nutrient needs, particularly nitrogen. Ensure the plants receive enough sunlight and space to grow properly. Regularly check for pests, such as aphids and soybean cyst nematodes, and control them promptly. Rotate crops to reduce the buildup of soil-borne diseases."
  },
  "Squash___Powdery_mildew": {
    "description": "Powdery mildew on squash is caused by *Erysiphe cichoracearum* and *Sphaerotheca fuliginea*, fungi that produce a white, powdery coating on the leaves and stems. The disease causes leaves to turn yellow and dry out, reducing the plant’s ability to photosynthesize. If left unchecked, it can stunt plant growth and lead to a poor harvest.",
    "solution": "To control powdery mildew, remove and destroy infected plant parts to reduce fungal spores. Apply fungicides, especially those containing sulfur or neem oil, to protect healthy leaves. Water at the base of the plant to keep foliage dry, and improve airflow around the plants by ensuring they are spaced properly. Rotate crops to avoid the accumulation of fungal spores in the soil."
  },
  "Strawberry___Leaf_scorch": {
    "description": "Leaf scorch is a condition in strawberries caused by excessive heat, drought, or nutrient imbalances. It results in the edges of the leaves turning brown and crispy. This condition weakens the plant, reducing its ability to produce healthy fruit. Prolonged leaf scorch can lead to poor fruit quality and reduced yields.",
    "solution": "To manage leaf scorch, provide consistent moisture to the plants, especially during hot periods, but avoid waterlogging the soil. Mulch around the base of the plants to retain moisture and reduce heat stress. Ensure the plants receive balanced fertilization, particularly with potassium, to help them cope with environmental stress. Regularly inspect plants for signs of nutrient deficiencies and address them promptly."
  },
  "Strawberry___healthy": {
    "description": "Healthy strawberry plants exhibit lush, green leaves and strong fruit development. The plants should be free from disease, pest damage, and environmental stress. Healthy strawberries produce sweet, flavorful fruit with a firm texture, with no signs of discoloration or deformation.",
    "solution": "To maintain healthy strawberry plants, plant them in well-drained soil with plenty of organic matter and ensure they are spaced properly. Water regularly, especially during dry spells, and apply a balanced fertilizer to support growth and fruit production. Mulch around the plants to retain moisture and prevent weed growth. Monitor for pests, such as aphids and slugs, and control them as needed."
  },
  "Tomato___Bacterial_spot": {
    "description": "Bacterial spot in tomatoes is caused by *Xanthomonas vesicatoria*, a bacterium that leads to water-soaked lesions on the leaves and stems. These lesions expand and turn yellow, causing the affected areas to die off. Bacterial spot can also infect the fruit, causing dark, sunken spots that significantly reduce the quality of the tomatoes. The disease can spread quickly in warm, wet conditions.",
    "solution": "To control bacterial spot, remove and destroy infected plant parts to prevent the spread of the disease. Apply copper-based bactericides or streptomycin to protect healthy plants, especially during rainy periods. Avoid overhead irrigation, as it can spread the bacteria, and space the plants adequately to improve airflow. Crop rotation with non-solanaceous plants can also reduce the risk of reinfection."
  },
  "Tomato___Early_blight": {
    "description": "Early blight, caused by *Alternaria solani*, is characterized by dark, concentric-ring lesions on the lower leaves of tomato plants. These lesions gradually expand, causing the leaves to yellow and die. The disease can spread to the stems and fruit, leading to poor growth and reduced yields. Early blight thrives in humid conditions and is common in warm weather.",
    "solution": "To manage early blight, apply fungicides containing chlorothalonil or mancozeb at the first sign of infection and continue treatments as needed. Remove and destroy infected plant material to reduce the spread of spores. Rotate crops to prevent the buildup of fungal spores in the soil, and prune plants to improve airflow. Avoid wetting the foliage when irrigating to reduce disease pressure."
  },
  "Tomato___Late_blight": {
    "description": "Late blight is caused by *Phytophthora infestans*, a fungus-like pathogen that produces water-soaked lesions on leaves, stems, and fruit. Infected leaves turn brown and curl, while fruit may develop dark, mushy spots. Late blight spreads rapidly, particularly during wet, cool weather, and can quickly devastate a tomato crop if not controlled.",
    "solution": "To control late blight, apply fungicides such as copper-based treatments or systemic fungicides like mefenoxam at the first signs of infection. Remove and destroy infected plant material immediately. Water the plants at the base to avoid wetting the foliage and reduce the spread of spores. Rotate crops to reduce the risk of reinfection, and use resistant tomato varieties if available."
  },
  "Tomato___Leaf_Mold": {
    "description": "Leaf mold in tomatoes is caused by the fungus *Cladosporium fulvum*, which infects the leaves, causing them to develop yellow, moldy spots. Infected leaves become covered with a grayish mold and eventually die off. The disease is most common in humid, poorly ventilated environments and can severely impact the plant’s ability to produce fruit.",
    "solution": "To manage leaf mold, increase ventilation around the plants by ensuring proper spacing and pruning to improve airflow. Remove infected leaves and destroy them to reduce the spread of spores. Apply fungicides containing copper or chlorothalonil, and avoid overhead irrigation. Mulch around the plants to retain moisture at the base but avoid wetting the foliage."
  },
  "Tomato___Septoria_leaf_spot": {
    "description": "Septoria leaf spot, caused by *Septoria lycopersici*, appears as small, dark lesions with a yellow halo on the lower leaves. These lesions gradually expand and lead to leaf yellowing and premature leaf drop. The disease weakens the plant and reduces photosynthesis, ultimately affecting fruit quality and yield.",
    "solution": "To control septoria leaf spot, remove and destroy infected leaves to prevent further spread. Apply fungicides containing chlorothalonil or copper-based treatments to protect healthy leaves. Water at the base of the plants to keep the foliage dry, and space plants properly to reduce humidity around the leaves. Crop rotation and using disease-resistant varieties can help minimize the disease's impact."
  },
  "Tomato___Spider_mites_Two-spotted_spider_mite": {
    "description": "Two-spotted spider mites, *Tetranychus urticae*, are tiny pests that feed on the underside of tomato leaves, causing yellowing, speckling, and eventual leaf drop. Heavy infestations lead to stunted growth, reduced fruit quality, and weakened plants. Spider mites thrive in hot, dry conditions and reproduce quickly, making them a challenging pest to control.",
    "solution": "To manage spider mites, regularly inspect plants for signs of mite damage, such as stippling or webbing on the leaves. Use miticides or insecticidal soaps to control mite populations, and release natural predators like ladybugs or predatory mites. Increase humidity around the plants by misting the leaves or using overhead irrigation to make the environment less favorable for the mites. Prune affected leaves to reduce mite populations."
  },
  "Tomato___Target_Spot": {
    "description": "Target spot in tomatoes is caused by *Corynespora cassiicola*, leading to dark, circular lesions with concentric rings that resemble a target. The lesions appear on the leaves and stems, causing premature leaf drop and reducing the plant’s ability to photosynthesize. This disease thrives in warm, humid conditions and can significantly affect yield if not managed.",
    "solution": "To control target spot, apply fungicides containing chlorothalonil or mancozeb at the first signs of the disease. Prune the plants to improve airflow and reduce humidity around the foliage. Remove infected leaves and debris to prevent further spread of the disease. Rotate crops to reduce the buildup of fungal spores in the soil, and use resistant tomato varieties when available."
  },
  "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
    "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted by whiteflies and causes yellowing and curling of the tomato leaves, stunted growth, and reduced fruit set. The virus weakens the plant, making it more susceptible to secondary infections. Infected plants often exhibit a noticeable upward curling of leaves and pale, deformed fruit.",
    "solution": "To manage TYLCV, control whitefly populations using insecticidal soaps or systemic insecticides. Remove and destroy infected plants to prevent further spread of the virus. Use floating row covers to protect healthy plants from whitefly infestations. Avoid planting tomatoes in areas with a history of TYLCV, and use resistant tomato varieties when possible."
  },
  "Tomato___Tomato_mosaic_virus": {
    "description": "Tomato mosaic virus (ToMV) is a viral disease that causes mottled or streaked patterns on the leaves and fruit. Infected plants show stunted growth, poor fruit production, and may exhibit leaf curling or necrosis. The virus is spread through contact with infected plants or tools and can reduce the overall quality of the tomato crop.",
    "solution": "To control tomato mosaic virus, remove and destroy infected plants immediately. Disinfect gardening tools regularly to avoid spreading the virus. Use resistant tomato varieties to reduce the risk of infection. Avoid working with wet plants, as this can facilitate the transmission of the virus. Ensure good plant spacing to reduce the spread of the virus between plants."
  },
  "Tomato___healthy": {
    "description": "Healthy tomato plants have robust, dark green leaves and strong stems. The plants should be free from disease and pest damage, with no signs of wilting, yellowing, or leaf drop. Healthy tomatoes produce abundant fruit that is well-shaped, firm, and free from deformities. The plants should have a vigorous growth habit, ample flower production, and strong root systems.",
    "solution": "To maintain tomatoes, provide them with full sunlight, at least 6-8 hours per day. Plant in well-drained, fertile soil with good organic content. Water at the base of the plants to keep the foliage dry and prevent fungal infections. Fertilize regularly with a balanced mix to support healthy growth and fruit production. Inspect the plants frequently for early signs of pests or diseases, and take prompt action if necessary."
  }  
}


# Initialize spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize Google Translate for translation
translator = Translator()


# Define the PyTorch model (adjust based on your architecture)
class PlantDiseaseCNN(torch.nn.Module):
    def __init__(self):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate the flattened size
        dummy_input = torch.zeros(1, 3, 256, 256)
        with torch.no_grad():
            dummy_output = self.pool(self.conv4(self.pool(self.conv3(self.pool(self.conv2(self.pool(self.conv1(dummy_input))))))))
        self.flattened_size = dummy_output.numel()
        self.fc1 = torch.nn.Linear(self.flattened_size, 512)
        self.fc2 = torch.nn.Linear(512, len(CLASS_LABELS))  # Number of classes

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = self.pool(torch.nn.functional.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantDiseaseCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Image preprocessing
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Model prediction function
def model_predict(img_path, model):
    image_tensor = preprocess_image(img_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        class_index = torch.argmax(probabilities, dim=1).item()
        class_name = CLASS_LABELS[class_index]
        crop, disease = class_name.split('___')
        return crop, disease

# Translate text
def translate_text(text, lang):
    return translator.translate(text, dest=lang).text

@app.route('/')
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_stream = io.BytesIO()
            file.save(file_stream)
            file_stream.seek(0)  # Reset stream position for later use

            crop, disease = model_predict(file_stream, model)
            result_key = f"{crop}___{disease}"
            description_data = DESCRIPTIONS.get(result_key, {"description": "No description available.", "solution": "No solution available."})

            description = description_data['description']
            solution = description_data['solution']

            lang = request.form.get('language', 'en')
            translated_description = translate_text(description, lang)
            translated_solution = translate_text(solution, lang)

            return render_template(
                'result.html',
                crop=crop,
                disease=disease,
                description=translated_description,
                solution=translated_solution,
                filename=filename
            )
    return "File not allowed."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
