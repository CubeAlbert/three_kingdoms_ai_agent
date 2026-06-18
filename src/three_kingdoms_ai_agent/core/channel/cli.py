"""CLI channel — stdin / stdout transport for MVP interactive use."""

from __future__ import annotations

from .base import AgentResponse, Channel, Message


class CliChannel(Channel):
    """Concrete :class:`Channel` that uses :func:`input` and :func:`print`.

    The simplest possible transport — reads one line from stdin per
    :meth:`receive` call and writes :attr:`AgentResponse.content` to stdout.

    Usage::

        channel = CliChannel()
        msg = channel.receive()           # blocks on input()
        channel.send(AgentResponse("Hi")) # prints "Hi"
    """

    def __init__(self, prompt: str = "> ") -> None:
        """Initialize the CLI channel.

        Parameters
        ----------
        prompt : str
            The prompt string shown to the user on each input line.
        """
        self._prompt = prompt

    # -- Channel interface ----------------------------------------------------

    def receive(self) -> Message:
        """Read one line from stdin.

        Returns
        -------
        Message
            The user's input, with :attr:`Message.content` set to the
            stripped line.  An empty line yields an empty-content
            ``Message`` (the orchestrator decides whether to skip it).

        Raises
        ------
        EOFError
            Propagated from :func:`input` when the user sends EOF
            (Ctrl+D / Ctrl+Z).  The caller should treat this as a
            shutdown signal.
        """
        try:
            line = input(self._prompt)
        except EOFError:
            # Re-raise so the orchestrator can exit the loop cleanly.
            raise
        return Message(content=line.strip())

    def send(self, response: AgentResponse) -> None:
        """Print the response content to stdout.

        Parameters
        ----------
        response : AgentResponse
            The text to display.  Only :attr:`AgentResponse.content` is
            printed; :attr:`AgentResponse.metadata` is ignored by the
            CLI channel.
        """
        print(response.content)
