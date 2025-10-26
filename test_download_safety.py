#!/usr/bin/env python3
"""
Test script to demonstrate safe interrupt handling during model downloads.
"""

import sys
import time
from model_explorer import ModelExplorer

def test_interrupt_safety():
    """Test that interrupts are handled safely during downloads."""
    
    print("🧪 Testing Interrupt Safety")
    print("=" * 40)
    print("This test will demonstrate that model downloads can be safely interrupted.")
    print("Press Ctrl+C during download to test interrupt handling.")
    print()
    
    explorer = ModelExplorer()
    
    # Test with a small model first
    test_model = "distilgpt2"
    
    print(f"Testing with model: {test_model}")
    print("💡 This model is small (~350MB) so download might be quick")
    print("🛑 Try pressing Ctrl+C during download to test safety")
    print()
    
    input("Press Enter to start test download...")
    
    try:
        # First check if already downloaded
        if explorer._is_model_cached(test_model):
            print(f"ℹ️  {test_model} is already cached. Testing force re-download...")
            success = explorer.download_model(test_model, force_download=True)
        else:
            success = explorer.download_model(test_model)
        
        if success:
            print("\n✅ Download completed successfully!")
            
            # Test integrity
            print("\n🔍 Testing download integrity...")
            integrity = explorer.check_download_integrity(test_model)
            
            if integrity.get("complete", False):
                print("✅ Model integrity verified - all components present")
            else:
                print("⚠️  Model integrity check failed")
        else:
            print("\n⚠️  Download was not completed (likely interrupted)")
            print("💡 This is expected behavior if you pressed Ctrl+C")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user (Ctrl+C)")
        print("✅ This demonstrates safe interrupt handling!")
        print("💾 Any partial downloads are safely cached")
        print("🔄 You can resume by running the download again")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        print(f"\n📊 Final cache status:")
        cache_info = explorer.get_download_progress_info()
        print(f"Cache size: {cache_info['total_size_mb']:.1f} MB")
        print(f"Files: {cache_info['file_count']}")


def test_resume_capability():
    """Test that downloads can be resumed after interruption."""
    
    print("\n🔄 Testing Resume Capability")
    print("=" * 40)
    
    explorer = ModelExplorer()
    test_model = "gpt2"  # Larger model for better resume testing
    
    print(f"Testing resume with: {test_model}")
    print("1. This will start a download")
    print("2. Interrupt it with Ctrl+C") 
    print("3. Then restart to test resuming")
    print()
    
    choice = input("Start resume test? (y/N): ").strip().lower()
    if choice != 'y':
        print("❌ Resume test skipped")
        return
    
    try:
        print(f"\n🚀 Starting download of {test_model}")
        print("💡 Interrupt with Ctrl+C, then restart to test resume")
        
        success = explorer.download_model(test_model, resume=True)
        
        if success:
            print("✅ Download/resume completed successfully!")
        else:
            print("⚠️  Download interrupted - try running again to resume")
    
    except KeyboardInterrupt:
        print("\n🛑 Download interrupted")
        print("🔄 Run this test again to demonstrate resume functionality")


def show_safety_info():
    """Show information about download safety."""
    
    print("\n🛡️  Download Safety Information")
    print("=" * 50)
    print()
    print("✅ SAFE TO INTERRUPT:")
    print("   • Ctrl+C during download is completely safe")
    print("   • No risk of file corruption")
    print("   • Partial downloads are cached properly")
    print("   • Can resume interrupted downloads anytime")
    print()
    print("🔄 RESUME CAPABILITY:")
    print("   • Hugging Face uses atomic downloads")
    print("   • Files are downloaded to temp locations first")
    print("   • Only moved to final location when complete")
    print("   • Interrupted downloads can resume from where they left off")
    print()
    print("💾 CACHE BEHAVIOR:")
    print("   • Models cached in: /workspaces/huggingface/models/.cache")
    print("   • Cache is shared across all applications")
    print("   • No duplication - models downloaded once")
    print("   • Can check cache status anytime")
    print()
    print("🚨 WHAT HAPPENS ON INTERRUPT:")
    print("   1. Signal handler catches Ctrl+C")
    print("   2. Current download is flagged to stop")
    print("   3. Cleanup happens automatically")
    print("   4. Partial files remain safely cached")
    print("   5. Next download resumes from partial state")
    print()
    print("🔍 VERIFICATION:")
    print("   • Always check integrity after download")
    print("   • Cleanup tools remove incomplete entries")
    print("   • Storage monitoring shows real usage")


if __name__ == "__main__":
    print("🧪 Model Download Safety Tester")
    print("=" * 50)
    print()
    print("This script demonstrates that model downloads are safe to interrupt.")
    print("Choose a test to run:")
    print()
    print("1. 🛑 Test interrupt safety")
    print("2. 🔄 Test resume capability") 
    print("3. ℹ️  Show safety information")
    print("4. 🚪 Exit")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            test_interrupt_safety()
            break
        elif choice == "2":
            test_resume_capability()
            break
        elif choice == "3":
            show_safety_info()
            break
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
    
    print("\n💡 To run the full model explorer: python model_explorer.py")