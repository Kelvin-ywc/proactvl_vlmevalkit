import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src.app.core.config import get_settings
from src.app.api.ws import router as ws_router
from src.app.api.hello import router as hello_router
from src.app.models.model import get_model

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ====== 启动时加载模型（可选预热）======
    model = get_model()
    # 如果实现了 warmup()，在后台线程里执行；没实现就跳过
    try:
        await run_in_threadpool(model.warmup)
    except AttributeError:
        pass
    except Exception as e:
        logging.exception("Model warmup failed: %s", e)
    yield
    # ====== 这里可做资源释放（如关闭会话/线程池）======

app = FastAPI(title="ProAct Backend", version="0.1.0", lifespan=lifespan)

# CORS 仅对 HTTP 有效，WS 需自行校验 Origin（见 ws.py 注释）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    # 增加一个简单标志，便于就绪探针观测
    try:
        get_model()  # 如果加载失败会抛
        return {"ok": True, "model_loaded": True}
    except Exception:
        return {"ok": True, "model_loaded": False}

# WebSocket 路由：/ws/stream
app.include_router(ws_router)
app.include_router(hello_router)