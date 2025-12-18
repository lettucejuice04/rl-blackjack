"""Simple Blackjack environment with Hi-Lo running count.

Observation: (player_sum, dealer_showing, usable_ace, true_count_bin)
Actions: 0=stick, 1=hit

This environment allows injecting a deterministic `deck` for tests.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple


def _hi_lo_value(card: int) -> int:
    # Hi-Lo system: 2-6 => +1, 7-9 => 0, 10 or Ace(1) => -1
    if 2 <= card <= 6:
        return 1
    if 7 <= card <= 9:
        return 0
    return -1


class BlackjackEnv:
    def __init__(self, n_decks: int = 1, seed: Optional[int] = None, deck: Optional[List[int]] = None):
        self.n_decks = n_decks
        self._seed = seed
        self._rand = random.Random(seed)
        self._initial_deck = list(deck) if deck is not None else None
        self.reset_deck()
        self.running_count = 0
        self.done = False

    def reset_deck(self) -> None:
        if self._initial_deck is not None:
            self.deck = list(self._initial_deck)
        else:
            base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
            self.deck = base * 4 * self.n_decks
            self._rand.shuffle(self.deck)

    def seed(self, s: int):
        self._rand.seed(s)

    def draw_card(self) -> int:
        if not self.deck:
            # reshuffle (simple behavior)
            self.reset_deck()
        return self.deck.pop()

    @staticmethod
    def _card_value(card: int) -> int:
        # card 1 is Ace.
        return card

    @staticmethod
    def _usable_ace(hand: List[int]) -> bool:
        return 1 in hand and sum(1 if c != 1 else 1 for c in hand) + 10 <= 21

    @staticmethod
    def _score(hand: List[int]) -> int:
        s = sum(hand)
        if 1 in hand and s + 10 <= 21:
            return s + 10
        return s

    def _binned_true_count(self) -> int:
        remaining = max(1, len(self.deck))
        remaining_decks = remaining / 52.0
        true_count = self.running_count / remaining_decks
        # Bin/clamp to reasonable integers for tabular methods
        return max(-10, min(10, int(round(true_count))))

    def true_count(self) -> float:
        """Return the raw (non-binned) True Count (running_count / remaining_decks).

        Useful for betting strategies and analysis where fractional TC matters.
        """
        remaining = max(1, len(self.deck))
        remaining_decks = remaining / 52.0
        return self.running_count / remaining_decks

    def reset(self) -> Tuple[int, int, bool, int]:
        self.reset_deck()
        self.running_count = 0
        self.done = False

        # draw initial hands
        self.player = [self.draw_card(), self.draw_card()]
        self.dealer = [self.draw_card(), self.draw_card()]

        # At the table only one dealer card is visible; only count seen cards
        # Count player's two cards and dealer's showing card (dealer[0])
        for c in self.player + [self.dealer[0]]:
            self.running_count += _hi_lo_value(c)

        return self._get_obs()

    def _get_obs(self) -> Tuple[int, int, bool, int]:
        player_score = self._score(self.player)
        dealer_show = self.dealer[0]
        usable = self._usable_ace(self.player)
        tc = self._binned_true_count()
        return (player_score, dealer_show, usable, tc)

    def step(self, action: int) -> Tuple[Tuple[int, int, bool, int], float, bool, dict]:
        assert action in (0, 1)
        if self.done:
            raise RuntimeError("Episode is done; call reset()")

        if action == 1:  # hit
            card = self.draw_card()
            self.player.append(card)
            self.running_count += _hi_lo_value(card)
            if self._score(self.player) > 21:
                self.done = True
                return self._get_obs(), -1.0, True, {}
            return self._get_obs(), 0.0, False, {}

        # action == 0 (stick) -> dealer plays
        # reveal dealer hidden card and count it
        self.running_count += _hi_lo_value(self.dealer[1])

        while self._score(self.dealer) < 17:
            c = self.draw_card()
            self.dealer.append(c)
            self.running_count += _hi_lo_value(c)

        player_score = self._score(self.player)
        dealer_score = self._score(self.dealer)

        self.done = True
        if player_score > 21:
            return self._get_obs(), -1.0, True, {}
        if dealer_score > 21 or player_score > dealer_score:
            return self._get_obs(), 1.0, True, {}
        if player_score == dealer_score:
            return self._get_obs(), 0.0, True, {}
        return self._get_obs(), -1.0, True, {}

    # Helper for tests
    def set_deck(self, deck: List[int]) -> None:
        self._initial_deck = list(deck)
        self.reset_deck()

    def __repr__(self) -> str:
        return f"BlackjackEnv(player={self.player}, dealer={self.dealer}, running_count={self.running_count})"
