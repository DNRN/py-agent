import os
import logfire

from zoneinfo import ZoneInfo
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

logfire.configure(service_name='pydantic-agent')
logfire.instrument_pydantic()
logfire.instrument_pydantic_ai()

model = OpenAIChatModel(
    'kimi-k2.6',
    provider=OpenAIProvider(
        base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.moonshot.ai/v1'),
        api_key=os.environ['OPENAI_API_KEY'],
    )
)

agent = Agent(model, name='time-agent')

@agent.tool
async def get_time(ctx, timezone: str | None = None) -> str:
    """Get the current time. optionally add a timezone"""
    from datetime import datetime
    now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
    return now.strftime('%H:%M %Z')

messages = []
while True:
    try:
        user_input = input('>')
    except (KeyboardInterrupt, EOFError):
        print()
        break
    if not user_input:
        continue
    user_input = user_input.strip()
    if user_input in {'quit', 'exit'}:
        break

    try:
        result = agent.run_sync(user_input, message_history=messages)
    except KeyboardInterrupt:
        print('interrupted')
        continue
    messages = result.all_messages()
    print(result.output)
