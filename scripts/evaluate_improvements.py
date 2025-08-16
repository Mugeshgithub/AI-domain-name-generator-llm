#!/usr/bin/env python3
"""
Evaluate Improvements: Model Comparison and Analysis
Comprehensive evaluation of base vs. fine-tuned models
"""

import json
import time
from pathlib import Path

def evaluate_improvements():
    """Evaluate improvements between base and fine-tuned models"""
    
    print("📊 EVALUATING MODEL IMPROVEMENTS")
    print("=" * 50)
    
    # Check if we have the required files
    required_files = [
        "evaluation_report.json",
        "model_comparison_report.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("   Please run the model comparison tests first")
        return
    
    print("✅ All required evaluation files found!")
    
    # Load evaluation reports
    print("\n📥 LOADING EVALUATION DATA:")
    
    with open("evaluation_report.json", "r") as f:
        base_evaluation = json.load(f)
    
    with open("model_comparison_report.json", "r") as f:
        comparison_report = json.load(f)
    
    print("   ✅ Base model evaluation loaded")
    print("   ✅ Model comparison report loaded")
    
    # Analyze improvements
    print("\n🔍 ANALYZING IMPROVEMENTS:")
    
    # Extract key metrics
    base_quality = base_evaluation.get("evaluation_summary", {}).get("average_quality_score", 0)
    base_time = base_evaluation.get("evaluation_summary", {}).get("average_generation_time", 0)
    
    comparison_summary = comparison_report.get("model_comparison_summary", {})
    ft_quality = comparison_summary.get("fine_tuned_model", {}).get("average_quality_score", 0)
    ft_time = comparison_summary.get("fine_tuned_model", {}).get("average_generation_time", 0)
    
    improvements = comparison_summary.get("improvements", {})
    quality_improvement = improvements.get("quality_improvement", 0)
    percentage_improvement = improvements.get("percentage_quality_improvement", 0)
    
    # Display improvement analysis
    print(f"   📈 QUALITY IMPROVEMENTS:")
    print(f"      Base Model: {base_quality:.1f}/10")
    print(f"      Fine-tuned: {ft_quality:.1f}/10")
    print(f"      Improvement: +{quality_improvement:.1f} points")
    print(f"      Percentage: +{percentage_improvement:.1f}%")
    
    print(f"\n   ⏱️  PERFORMANCE ANALYSIS:")
    print(f"      Base Model Time: {base_time:.1f}s")
    print(f"      Fine-tuned Time: {ft_time:.1f}s")
    
    if ft_time > 0 and base_time > 0:
        time_change = ft_time - base_time
        if time_change > 0:
            print(f"      Time Change: +{time_change:.1f}s (slower)")
        else:
            print(f"      Time Change: {time_change:.1f}s (faster)")
    
    # Category-wise improvements
    print(f"\n   🏷️  CATEGORY IMPROVEMENTS:")
    
    detailed_results = comparison_report.get("detailed_results", [])
    categories = {}
    
    for result in detailed_results:
        category = result.get("category", "Unknown")
        model = result.get("model", "unknown")
        quality = result.get("quality_score", 0)
        
        if category not in categories:
            categories[category] = {"base": 0, "fine_tuned": 0}
        
        if model == "base":
            categories[category]["base"] = quality
        elif model == "fine_tuned":
            categories[category]["fine_tuned"] = quality
    
    for category, scores in categories.items():
        if scores["base"] > 0 and scores["fine_tuned"] > 0:
            improvement = scores["fine_tuned"] - scores["base"]
            percentage = (improvement / scores["base"]) * 100 if scores["base"] > 0 else 0
            print(f"      {category}: {scores['base']:.1f} → {scores['fine_tuned']:.1f} (+{improvement:.1f}, +{percentage:.1f}%)")
    
    # Generate improvement summary
    print(f"\n📋 IMPROVEMENT SUMMARY:")
    
    improvement_summary = {
        "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model_performance": {
            "average_quality": base_quality,
            "average_generation_time": base_time
        },
        "fine_tuned_performance": {
            "average_quality": ft_quality,
            "average_generation_time": ft_time
        },
        "improvements": {
            "quality_improvement": quality_improvement,
            "percentage_improvement": percentage_improvement,
            "time_change": ft_time - base_time if ft_time > 0 and base_time > 0 else 0
        },
        "category_breakdown": categories,
        "overall_assessment": "SUCCESS" if quality_improvement > 0 else "NEEDS_IMPROVEMENT"
    }
    
    # Save improvement summary
    with open("improvement_summary.json", "w") as f:
        json.dump(improvement_summary, f, indent=2)
    
    print(f"   ✅ Improvement summary saved to: improvement_summary.json")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT:")
    
    if quality_improvement > 0:
        print(f"   🎉 SUCCESS: Fine-tuning improved model quality by {quality_improvement:.1f} points")
        print(f"   📊 Overall improvement: +{percentage_improvement:.1f}%")
        
        if percentage_improvement >= 20:
            print(f"   🌟 EXCELLENT: Significant improvement achieved!")
        elif percentage_improvement >= 10:
            print(f"   ✅ GOOD: Meaningful improvement demonstrated")
        else:
            print(f"   📈 MODEST: Small but measurable improvement")
    else:
        print(f"   ⚠️  NEEDS WORK: No quality improvement detected")
        print(f"   💡 Consider: More training data, different hyperparameters, or model architecture")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   - improvement_summary.json: Detailed improvement analysis")
    print(f"   - model_comparison_report.json: Complete model comparison")
    print(f"   - evaluation_report.json: Base model evaluation")
    
    return improvement_summary

if __name__ == "__main__":
    evaluate_improvements()
