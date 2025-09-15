# telegram_bot.py
import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, 
    ContextTypes, ConversationHandler, filters
)
from agents import agents_system

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot token from BotFather
BOT_TOKEN = "8427368845:AAFxETlP6UsgeRWyc1eme9VmF6_SByP9GgQ"  # Replace with your actual bot token

# Conversation states
SETTING_ROLE, SETTING_INTERESTS, CHATTING = range(3)

# User data storage (in production, use a database)
user_profiles = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send welcome message and start profile setup."""
    user_id = update.effective_user.id
    welcome_text = """
    🤖 Welcome to AetherNet! 
    I'm your AI professional ecosystem assistant.
    
    Let's set up your profile first so I can provide personalized help.
    """
    
    await update.message.reply_text(welcome_text)
    await update.message.reply_text("What's your current role? (e.g., Software Engineer, Marketing Associate)")
    
    return SETTING_ROLE

async def set_role(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store the user's role and ask for interests."""
    user_id = update.effective_user.id
    role = update.message.text
    
    # Initialize user profile if not exists
    if user_id not in user_profiles:
        user_profiles[user_id] = {}
    
    user_profiles[user_id]['role'] = role
    
    await update.message.reply_text("Great! What are your career interests? (e.g., leadership, data science, design)")
    
    return SETTING_INTERESTS

async def set_interests(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store the user's interests and start main conversation."""
    user_id = update.effective_user.id
    interests = update.message.text
    
    user_profiles[user_id]['interests'] = interests
    
    # Create keyboard with quick actions
    reply_keyboard = [
        ["/onboarding", "/learning"],
        ["/career", "/help"]
    ]
    
    await update.message.reply_text(
        "Profile setup complete! 🎉\n\n"
        "You can now ask me anything about:\n"
        "• 🎯 Onboarding & company info\n"
        "• 📚 Learning & skill development\n"
        "• 🚀 Career growth & coaching\n\n"
        "Or use the quick buttons below:",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, resize_keyboard=True)
    )
    
    return CHATTING

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    user_id = update.effective_user.id
    user_query = update.message.text
    
    # Check if user has profile
    if user_id not in user_profiles:
        await update.message.reply_text("Please start with /start to set up your profile first.")
        return
    
    # Show typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    # Process the query through our agent system
    try:
        response_data = agents_system.process_query(user_query, user_profiles[user_id])
        
        # Build response message
        reply_message = f"{response_data['agent_name']}:\n\n{response_data['answer']}"
        
        # Add source information (if available)
        if response_data['sources']:
            unique_sources = list(set(response_data['sources']))
            sources_text = "\n\n📁 Information Sources:\n" + "\n".join([f"• {src}" for src in unique_sources[:3]])  # Show first 3 sources
            if len(unique_sources) > 3:
                sources_text += f"\n• ... and {len(unique_sources)} more documents"
            reply_message += sources_text
        
        await update.message.reply_text(reply_message)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        await update.message.reply_text("Sorry, I encountered an error processing your request. Please try again later.")

async def quick_onboarding(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick action for onboarding questions."""
    await update.message.reply_text("What would you like to know about onboarding, company policies, or your team?")

async def quick_learning(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick action for learning questions."""
    await update.message.reply_text("What would you like to learn or which skills would you like to develop?")

async def quick_career(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Quick action for career questions."""
    await update.message.reply_text("What career guidance are you looking for? (goals, reviews, advancement)")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    help_text = """
    🤖 AetherNet Help:
    
    /start - Set up your profile
    /onboarding - Ask about onboarding
    /learning - Ask about learning
    /career - Ask about career growth
    /help - Show this help message
    
    You can also just type your questions directly!
    """
    await update.message.reply_text(help_text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text(
        "Goodbye! You can always come back with /start",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Set up conversation handler with states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SETTING_ROLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_role)],
            SETTING_INTERESTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_interests)],
            CHATTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
                CommandHandler("onboarding", quick_onboarding),
                CommandHandler("learning", quick_learning),
                CommandHandler("career", quick_career),
                CommandHandler("help", help_command),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Add handlers
    application.add_handler(conv_handler)
    
    # Start the Bot
    print("🤖 AetherNet Telegram Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()