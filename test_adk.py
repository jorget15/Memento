#!/usr/bin/env python3
"""
Example script showing how to use Google ADK properly
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_adk_import():
    """Test if Google ADK can be imported"""
    try:
        import google.adk as adk
        print("✅ Google ADK imported successfully!")
        print(f"📦 Version: {adk.__version__}")
        print(f"📋 Available components: {dir(adk)}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Google ADK: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with Google ADK: {e}")
        return False

def create_simple_agent():
    """Create a simple ADK agent"""
    try:
        from google.adk import Agent
        
        # Check if API key is available
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("⚠️  No GOOGLE_API_KEY found in environment variables")
            print("📝 Please add your API key to the .env file")
            return None
        
        print("🚀 Creating ADK Agent...")
        agent = Agent(
            model="gemini-1.5-flash",
            api_key=api_key
        )
        print("✅ Agent created successfully!")
        return agent
        
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        return None

def main():
    """Main function to test ADK functionality"""
    print("🔍 Testing Google ADK...")
    print("=" * 50)
    
    # Test import
    if not test_adk_import():
        return
    
    print("\n" + "=" * 50)
    
    # Test agent creation
    agent = create_simple_agent()
    if agent:
        print("🎉 ADK is working correctly!")
        
        # Optional: Test a simple query
        try:
            response = agent.run("Hello! What is Google ADK?")
            print(f"🤖 Agent response: {response}")
        except Exception as e:
            print(f"⚠️  Agent query failed: {e}")
    else:
        print("❌ ADK agent creation failed")

if __name__ == "__main__":
    main()