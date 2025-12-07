from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaMessage,
        BetaTextBlock,
        BetaUsage,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

async def test_container_persistence(allow_model_requests: None):
    # Mock response with container ID
    # We need to mock the response object to have 'container' attribute
    # Since BetaMessage is a Pydantic model, we can't easily add attributes if they are not defined.
    # But we can use a custom mock or rely on the fact that we updated AnthropicModel to check hasattr.
    
    # Create a mock message that has a container attribute
    class MockMessageWithContainer(BetaMessage):
        container: Any = None

    c = MockMessageWithContainer(
        id='123',
        content=[BetaTextBlock(text='response', type='text')],
        model='claude-3-5-sonnet-20241022',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=BetaUsage(input_tokens=10, output_tokens=5),
        container=type('Container', (), {'id': 'container-123'}),
    )
    
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('test prompt')
    
    # Verify container ID is stored in provider_details
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    assert last_message.provider_details['anthropic_container_id'] == 'container-123'

    # Now run again passing the container ID
    # We need to manually pass it in model_settings as per our implementation
    # The user (or agent loop) needs to pass it.
    
    mock_client.index = 0
    await agent.run(
        'next prompt', 
        model_settings=AnthropicModelSettings(anthropic_container={'id': 'container-123'})
    )
    
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[1]
    assert completion_kwargs['container'] == {'id': 'container-123'}
