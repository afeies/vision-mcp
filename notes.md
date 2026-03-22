- software development kit (SDK): collection of tools, libraries, and documentation that helps developers build applications
- FastMCP: a lightweight framework for building MCP servers in Python
- mcp.run(): starts the server, listens over   
  stdio, reads JSON-RPC requests from stdin, and writes 
  responses to stdout
- JSON-RPC: lightweight remote procedural call protocol that uses JSON  
- __name__ == "__main__": server only   
  starts when you run the file directly (python 
  opencv_mcp_server.py), not when it's imported

- @mcp.tool(): Python decorator that registers the function below it as a tool the MCP server exposes to clients
    - when the server starts, collects all functions and advertises them to connected clients (like Claude Desktop)