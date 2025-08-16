#!/usr/bin/env python3
"""
Create Training Dataset for Domain Name Generation
Synthetic dataset creation with diverse business types
"""

import json
import random

def create_training_dataset():
    """Create synthetic training dataset for domain name generation"""
    
    print("🏗️  CREATING SYNTHETIC TRAINING DATASET")
    print("=" * 50)
    
    # Business categories with varying complexity
    business_categories = [
        {
            "category": "Food & Beverage",
            "examples": [
                "organic coffee shop downtown",
                "vegan restaurant specializing in plant-based cuisine",
                "craft brewery with taproom",
                "artisanal bakery with gluten-free options",
                "farm-to-table restaurant with seasonal menu"
            ]
        },
        {
            "category": "Technology",
            "examples": [
                "AI software startup for business automation",
                "mobile app development company",
                "cybersecurity consulting firm",
                "cloud computing solutions provider",
                "blockchain technology consulting"
            ]
        },
        {
            "category": "Health & Wellness",
            "examples": [
                "yoga studio with meditation classes",
                "organic skincare products",
                "personal fitness training service",
                "holistic wellness center",
                "mental health counseling practice"
            ]
        },
        {
            "category": "Professional Services",
            "examples": [
                "digital marketing agency for small businesses",
                "wedding photography studio",
                "pet grooming salon and spa",
                "online bookstore with rare books",
                "sustainable fashion brand"
            ]
        }
    ]
    
    # Generate domain name outputs for each business description
    training_data = []
    
    for category in business_categories:
        print(f"📊 Processing {category['category']} examples...")
        
        for business_desc in category['examples']:
            # Create instruction-output pair
            instruction = f"Generate 3 domain names for: {business_desc}"
            
            # Generate sample domain names (these would normally come from the model)
            # For now, we'll create realistic examples
            if "coffee" in business_desc.lower():
                output = "organicbeanscafe.com, downtowncoffee.org, freshbreworganic.net"
            elif "ai" in business_desc.lower() or "software" in business_desc.lower():
                output = "aiautomation.com, businessai.net, aistartup.com"
            elif "yoga" in business_desc.lower():
                output = "yogameditation.com, yogastudio.net, meditationyoga.com"
            elif "marketing" in business_desc.lower():
                output = "digimarketing.com, smallbizads.net, marketboost.com"
            elif "photography" in business_desc.lower():
                output = "weddingphoto.com, bridalcapture.net, matrimonyphoto.com"
            elif "pet" in business_desc.lower():
                output = "petspa.com, groompro.net, furryfriends.com"
            elif "bookstore" in business_desc.lower():
                output = "rarebooks.com, bookfinder.net, literarytreasure.com"
            elif "fashion" in business_desc.lower():
                output = "ecofashion.com, greenstyle.net, sustainablewear.com"
            elif "brewery" in business_desc.lower():
                output = "craftbrew.com, taproom.net, beerartisan.com"
            elif "skincare" in business_desc.lower():
                output = "organicskin.com, naturalglow.net, purebeauty.com"
            else:
                # Generic fallback
                words = business_desc.lower().split()
                domain1 = f"{words[0]}{words[1]}.com"
                domain2 = f"{words[1]}{words[0]}.net"
                domain3 = f"{words[0]}pro.com"
                output = f"{domain1}, {domain2}, {domain3}"
            
            training_example = {
                "instruction": instruction,
                "output": output,
                "category": category['category'],
                "complexity": "medium" if len(business_desc.split()) > 4 else "low"
            }
            
            training_data.append(training_example)
    
    # Save training dataset
    output_file = "training_dataset.json"
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"\n✅ TRAINING DATASET CREATED SUCCESSFULLY!")
    print(f"   Total examples: {len(training_data)}")
    print(f"   Categories: {len(business_categories)}")
    print(f"   Output file: {output_file}")
    
    # Show sample data
    print(f"\n📋 SAMPLE TRAINING DATA:")
    for i, example in enumerate(training_data[:3]):
        print(f"   Example {i+1}:")
        print(f"     Instruction: {example['instruction']}")
        print(f"     Output: {example['output']}")
        print(f"     Category: {example['category']}")
        print()
    
    return training_data

if __name__ == "__main__":
    create_training_dataset()
