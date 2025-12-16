"""
Unit tests for daily_runner.py

测试策略：使用 Mock 隔离外部依赖 (Telegram API, DuckDB, data_loader)
确保核心逻辑的正确性。
"""

import pytest
from unittest.mock import patch, MagicMock
import polars as pl
from datetime import date
import sys
import os

# 确保能导入 daily_runner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGetAllTickers:
    """测试 get_all_tickers 函数"""

    def test_flatten_tickers(self):
        """验证 TICKERS 字典被正确展平为列表"""
        from daily_runner import get_all_tickers

        with patch("daily_runner.TICKERS", {"US": ["SPY", "AAPL"], "CN": ["600519"]}):
            result = get_all_tickers()
            assert isinstance(result, list)
            assert len(result) == 3
            assert "SPY" in result
            assert "AAPL" in result
            assert "600519" in result

    def test_empty_tickers(self):
        """空的 TICKERS 字典应返回空列表"""
        from daily_runner import get_all_tickers

        with patch("daily_runner.TICKERS", {}):
            result = get_all_tickers()
            assert result == []


class TestEnsureStrategiesRegistered:
    """测试 ensure_strategies_registered 函数"""

    def test_registers_default_when_empty(self):
        """当注册表为空时，应注册默认策略"""
        from daily_runner import ensure_strategies_registered

        mock_registry: dict = {}
        with (
            patch("daily_runner.STRATEGY_REGISTRY", mock_registry),
            patch("daily_runner.simple_ma_strategy", lambda x: x) as mock_strategy,
        ):
            ensure_strategies_registered()
            assert "SimpleMA" in mock_registry

    def test_does_nothing_when_not_empty(self):
        """当注册表非空时，不应修改"""
        from daily_runner import ensure_strategies_registered

        existing_strategy = MagicMock()
        mock_registry = {"CustomStrategy": existing_strategy}

        with patch("daily_runner.STRATEGY_REGISTRY", mock_registry):
            ensure_strategies_registered()
            # 只应有原始策略，不应添加 SimpleMA
            assert len(mock_registry) == 1
            assert "CustomStrategy" in mock_registry


class TestSendTelegram:
    """测试 send_telegram 函数"""

    def test_skip_when_no_token(self, capsys):
        """无 TG_TOKEN 时应跳过发送"""
        from daily_runner import send_telegram

        with patch("daily_runner.TG_TOKEN", ""):
            send_telegram("Test message")
            captured = capsys.readouterr()
            assert "SKIPPING TELEGRAM SEND (TG_TOKEN missing)" in captured.out

    def test_skip_when_no_chat_id(self, capsys):
        """有 TG_TOKEN 但无 TG_CHAT_ID 时应跳过发送"""
        from daily_runner import send_telegram

        with (
            patch("daily_runner.TG_TOKEN", "fake_token"),
            patch.dict(os.environ, {"TG_CHAT_ID": ""}, clear=False),
            patch("os.getenv", return_value=None),
        ):
            send_telegram("Test message")
            captured = capsys.readouterr()
            assert "SKIPPING TELEGRAM SEND" in captured.out

    def test_successful_send(self):
        """正常发送消息"""
        from daily_runner import send_telegram

        with (
            patch("daily_runner.TG_TOKEN", "fake_token"),
            patch("os.getenv", return_value="123456789"),
            patch("daily_runner.requests.post") as mock_post,
        ):
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            send_telegram("Test message")

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "fake_token" in call_args[0][0]
            assert call_args[1]["json"]["text"] == "Test message"
            assert call_args[1]["json"]["chat_id"] == "123456789"

    def test_handles_request_exception(self, capsys):
        """请求异常时应捕获并打印错误"""
        from daily_runner import send_telegram

        with (
            patch("daily_runner.TG_TOKEN", "fake_token"),
            patch("os.getenv", return_value="123456789"),
            patch("daily_runner.requests.post", side_effect=Exception("Network error")),
        ):
            send_telegram("Test message")
            captured = capsys.readouterr()
            assert "Failed to send Telegram message" in captured.out


class TestRunDailyJob:
    """测试 run_daily_job 主函数"""

    @pytest.fixture
    def mock_engine(self):
        """创建 Mock 的 BacktestEngine"""
        mock = MagicMock()
        # 返回包含足够数据的 DataFrame
        mock.run.return_value = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["SPY", "SPY"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000.0, 1100000.0],
                "signal": [0, 1],  # 从 HOLD -> BUY (触发 Flip + Buy)
                "market_return": [0.0, 0.01],
                "holding": [0, 0],
                "strategy_return": [0.0, 0.0],
                "equity_curve": [1.0, 1.0],
            }
        )
        return mock

    @pytest.fixture
    def mock_dependencies(self, mock_engine):
        """统一 Mock 所有外部依赖"""
        patches = [
            patch("daily_runner.sync_data"),
            patch("daily_runner.sync_fx_rates"),
            patch("daily_runner.BacktestEngine", return_value=mock_engine),
            patch("daily_runner.send_telegram"),
            patch("daily_runner.TICKERS", {"US": ["SPY"]}),  # 减少测试标的，加快速度
            patch(
                "daily_runner.STRATEGY_REGISTRY", {"TestMA": lambda x: x}
            ),  # 提供非空注册表
        ]

        started = [p.start() for p in patches]
        yield dict(
            zip(
                [
                    "sync_data",
                    "sync_fx",
                    "engine_cls",
                    "telegram",
                    "tickers",
                    "registry",
                ],
                started,
            )
        )
        for p in patches:
            p.stop()

    def test_full_flow_with_alert(self, mock_dependencies, capsys):
        """完整流程测试：数据更新 → 策略执行 → 生成告警"""
        from daily_runner import run_daily_job

        run_daily_job()

        # 验证数据同步被调用
        mock_dependencies["sync_data"].assert_called_once_with(["SPY"])
        mock_dependencies["sync_fx"].assert_called_once()

        # 验证 Telegram 发送被调用 (因为有 Flip + Buy 触发)
        mock_dependencies["telegram"].assert_called_once()
        call_args = mock_dependencies["telegram"].call_args[0][0]
        assert "SPY" in call_args
        assert "TestMA" in call_args

        captured = capsys.readouterr()
        assert "Daily Runner Start" in captured.out
        assert "Daily Runner Complete" in captured.out

    def test_no_alert_when_no_signal_change(self, mock_dependencies):
        """信号无变化且非买入时，不应发送告警"""
        from daily_runner import run_daily_job

        # 修改 engine 返回：信号一直是 HOLD
        mock_dependencies["engine_cls"].return_value.run.return_value = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["SPY", "SPY"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000000.0, 1100000.0],
                "signal": [0, 0],  # 一直 HOLD，无变化
                "market_return": [0.0, 0.01],
                "holding": [0, 0],
                "strategy_return": [0.0, 0.0],
                "equity_curve": [1.0, 1.0],
            }
        )

        run_daily_job()

        # 信号无变化，不应触发 Telegram
        mock_dependencies["telegram"].assert_not_called()

    def test_skips_ticker_with_insufficient_data(self, mock_dependencies, capsys):
        """数据不足时应跳过该标的"""
        from daily_runner import run_daily_job

        # 只返回 1 行数据 (需要至少 2 行来比较昨今信号)
        mock_dependencies["engine_cls"].return_value.run.return_value = pl.DataFrame(
            {
                "date": [date(2024, 1, 1)],
                "ticker": ["SPY"],
                "close": [100.0],
                "signal": [1],
            }
        )

        run_daily_job()

        captured = capsys.readouterr()
        assert "[Skip] Not enough data for SPY" in captured.out
        mock_dependencies["telegram"].assert_not_called()

    def test_handles_engine_exception(self, mock_dependencies, capsys):
        """引擎运行异常时应捕获并继续处理其他标的"""
        from daily_runner import run_daily_job

        mock_dependencies["engine_cls"].return_value.run.side_effect = Exception(
            "DB connection failed"
        )

        run_daily_job()

        captured = capsys.readouterr()
        assert "[Error] SPY: DB connection failed" in captured.out
        # 即使报错，也应完成整个流程
        assert "Daily Runner Complete" in captured.out


class TestAlertGeneration:
    """测试告警生成逻辑的边界情况"""

    def test_buy_signal_triggers_alert(self):
        """买入信号 (signal=1) 应触发告警"""
        from daily_runner import run_daily_job

        mock_engine = MagicMock()
        mock_engine.run.return_value = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["TEST", "TEST"],
                "close": [100.0, 101.0],
                "signal": [1, 1],  # 连续买入 (今天仍是买入)
            }
        )

        with (
            patch("daily_runner.sync_data"),
            patch("daily_runner.sync_fx_rates"),
            patch("daily_runner.BacktestEngine", return_value=mock_engine),
            patch("daily_runner.send_telegram") as mock_telegram,
            patch("daily_runner.TICKERS", {"TEST": ["TEST"]}),
            patch("daily_runner.STRATEGY_REGISTRY", {"TestStrategy": lambda x: x}),
        ):
            run_daily_job()

            # 虽然信号无变化 (1->1)，但 is_buy=True 仍应触发告警
            mock_telegram.assert_called_once()
            msg = mock_telegram.call_args[0][0]
            assert "🟢Buy" in msg

    def test_sell_flip_triggers_alert(self):
        """从买入翻转到卖出应触发告警"""
        from daily_runner import run_daily_job

        mock_engine = MagicMock()
        mock_engine.run.return_value = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "ticker": ["TEST", "TEST"],
                "close": [100.0, 95.0],
                "signal": [1, -1],  # BUY -> SELL (Flip, 但非 Buy)
            }
        )

        with (
            patch("daily_runner.sync_data"),
            patch("daily_runner.sync_fx_rates"),
            patch("daily_runner.BacktestEngine", return_value=mock_engine),
            patch("daily_runner.send_telegram") as mock_telegram,
            patch("daily_runner.TICKERS", {"TEST": ["TEST"]}),
            patch("daily_runner.STRATEGY_REGISTRY", {"TestStrategy": lambda x: x}),
        ):
            run_daily_job()

            mock_telegram.assert_called_once()
            msg = mock_telegram.call_args[0][0]
            assert "🔄Flip" in msg
            assert "SELL" in msg
