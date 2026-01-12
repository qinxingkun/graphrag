import json
import uuid
from graphrag.config import AppConfig,Neo4jConfig, PostgreSQLConfig, LLMConfig, MilvusConfig
from graphrag.service import Neo4jQueryService

class Neo4jQueryApp:
    
    def __init__(self, config: AppConfig = None):
        
        self.config = config or DEFAULT_CONFIG
        self.service = Neo4jQueryService(self.config)
        self.current_session = None
    
    def run_query(self, question: str, session_id: str = None) -> dict:
        
        result = self.service.query(question, session_id)
        
        if result["tool_calls"] and self.config.verbose:
            print("🔧 工具调用:")
            for i, tc in enumerate(result["tool_calls"], 1):
                print(f"  {i}. {tc['tool']}")
                print(f"     {json.dumps(tc['args'], ensure_ascii=False, indent=6)}")
            print()
        
        if self.config.verbose and result["conversation"]:
            for conv in result["conversation"]:
                emoji = {"user": "👤", "assistant": "🤖", "tool": "⚙️"}
                role_name = conv['role'].upper()
                print(f"{emoji.get(conv['role'], '•')} {role_name}: {conv['content']}\n")
        
        print(f"✅ 最终答案:\n{result['answer']}")
        
        return result
    
    def run_interactive(self):
        print("\n📚 命令列表:")
        print("  /new          - 开始新会话")
        print("  /history      - 查看当前会话历史")
        print("  /sessions     - 列出所有会话")
        print("  /load <id>    - 切换到指定会话")
        print("  /delete <id>  - 删除指定会话")
        print("  /stats        - 查看系统统计")
        print("  exit          - 退出系统")
        print("="*80 + "\n")
        
        self.current_session = str(uuid.uuid4())[:8]
        print(f"✨ 当前会话: {self.current_session}\n")
        
        while True:
            try:
                question = input("💭 > ").strip()
                
                print("user's question:",question)
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 再见!")
                    break
                
                if question.startswith('/'):
                    self._handle_command(question)
                    continue
                
                self.run_query(question, self.current_session)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {str(e)}\n")
    
    def _handle_command(self, command: str):
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/new':
            self.current_session = str(uuid.uuid4())[:8]
        
        elif cmd == '/history':
            history = self.service.get_conversation_history(self.current_session)
            if history:
                print(f"\n📜 会话历史 ({self.current_session}):")
                print("-" * 80)
                for msg in history:
                    role = msg['role'].upper()
                    print(f"{role}: {msg['content']}")
                    print(f"时间: {msg['created_at']}\n")
            else:
                print("\n暂无历史记录\n")
        
        elif cmd == '/sessions':
            sessions = self.service.list_sessions()
            if sessions:
                print("\n📋 所有会话:")
                print("-" * 80)
                for i, session in enumerate(sessions, 1):
                    print(f"{i}. ID: {session['session_id']}")
                    print(f"   消息数: {session['message_count']}")
                    print(f"   创建时间: {session['created_at']}")
                    print(f"   更新时间: {session['updated_at']}\n")
            else:
                print("\n暂无会话\n")
        
        elif cmd == '/load' and len(parts) > 1:
            self.current_session = parts[1]
            print(f"\n✅ 已切换到会话: {self.current_session}\n")
        
        elif cmd == '/delete' and len(parts) > 1:
            session_id = parts[1]
            self.service.delete_session(session_id)
            print(f"\n✅ 已删除会话: {session_id}\n")
            if session_id == self.current_session:
                self.current_session = str(uuid.uuid4())[:8]
                print(f"✨ 新会话: {self.current_session}\n")
        
        elif cmd == '/stats':
            if self.service.vector_store:
                stats = self.service.vector_store.get_stats()
                print("\n📊 系统统计:")
                print("-" * 80)
                print(f"向量数据库: {stats['collection_name']}")
                print(f"向量数量: {stats['total_count']}")
                print(f"向量维度: {stats['dimension']}")
                print()
            else:
                print("\n向量存储未启用\n")
        
        else:
            print("\n❌ 未知命令\n")

if __name__ == "__main__":
    
    config = AppConfig(
        neo4j=Neo4jConfig(),
        postgresql=PostgreSQLConfig(),
        milvus=MilvusConfig(),
        llm=LLMConfig(),
        verbose=True,
        enable_memory=True,
        enable_embedding=True
    )
    
    app = Neo4jQueryApp(config)
    
    app.run_interactive()