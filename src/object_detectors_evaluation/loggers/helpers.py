from __future__ import annotations

from .loguru import logger


def log_breaking_point(
    log,
    msg: str,
    n_top: int = 0,
    n_bottom: int = 0,
    top_char: str = " ",
    bottom_char: str = " ",
    fill_char: str = " ",
    num_chars: int = 100,
) -> None:
    top_line = top_char * num_chars
    bottom_line = bottom_char * num_chars

    for _ in range(n_top):
        log.info(top_line)

    log.info(msg.center(num_chars, fill_char))

    for _ in range(n_bottom):
        log.info(bottom_line)


if __name__ == "__main__":
    log_breaking_point(
        logger,
        msg="Test message for `log_breaking_point`",
        n_top=1,
        n_bottom=1,
        top_char="=",
        bottom_char="-",
        fill_char=" ",
        num_chars=80,
    )
    log_breaking_point(
        logger,
        msg="Another test message for `log_breaking_point`",
        n_top=2,
        n_bottom=2,
        top_char="*",
        bottom_char="*",
        fill_char="-",
        num_chars=80,
    )
