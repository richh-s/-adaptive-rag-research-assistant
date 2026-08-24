import { useEffect, useState } from 'react'
import {
  deleteConversation,
  listConversations,
  type ConversationSummaryInfo,
} from '../api/client'
import './ConversationsPanel.css'

interface ConversationsPanelProps {
  open: boolean
  onClose: () => void
  /** Load this conversation into the chat view. */
  onOpenConversation: (id: string) => void
  /** The conversation currently shown in the chat view, if any. */
  activeConversationId: string | null
  /** Called after a delete so the parent can reset the view if it was showing it. */
  onDeleted: (id: string) => void
}

function formatWhen(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000)
  const sameDay = new Date().toDateString() === date.toDateString()
  return sameDay
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

export function ConversationsPanel({
  open,
  onClose,
  onOpenConversation,
  activeConversationId,
  onDeleted,
}: ConversationsPanelProps) {
  const [conversations, setConversations] = useState<ConversationSummaryInfo[]>([])
  const [loadError, setLoadError] = useState(false)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    listConversations()
      .then((rows) => {
        if (!cancelled) {
          setConversations(rows)
          setLoadError(false)
        }
      })
      .catch(() => {
        if (!cancelled) setLoadError(true)
      })
    return () => {
      cancelled = true
    }
  }, [open])

  async function handleDelete(id: string) {
    await deleteConversation(id).catch(() => {})
    setConversations((prev) => prev.filter((c) => c.id !== id))
    onDeleted(id)
  }

  if (!open) return null

  return (
    <div className="corpus-overlay" onClick={onClose}>
      <aside
        className="corpus-drawer"
        onClick={(e) => e.stopPropagation()}
        aria-label="Conversation history"
      >
        <div className="corpus-drawer-header">
          <h2>Conversations</h2>
          <button type="button" className="corpus-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <p className="corpus-subtitle">
          Every conversation is saved on the server. Open one to continue where it left off.
        </p>

        {loadError && <p className="conversations-empty">Could not load conversations.</p>}
        {!loadError && conversations.length === 0 && (
          <p className="conversations-empty">No conversations yet — ask something to start one.</p>
        )}

        <ul className="conversation-list">
          {conversations.map((conversation) => (
            <li
              key={conversation.id}
              className={`conversation-item ${
                conversation.id === activeConversationId ? 'conversation-active' : ''
              }`}
            >
              <button
                type="button"
                className="conversation-open"
                onClick={() => {
                  onOpenConversation(conversation.id)
                  onClose()
                }}
              >
                <span className="conversation-title">{conversation.title}</span>
                <span className="conversation-meta">
                  {Math.floor(conversation.message_count / 2)} turn
                  {conversation.message_count >= 4 ? 's' : ''} · {formatWhen(conversation.updated_at)}
                </span>
              </button>
              <button
                type="button"
                className="conversation-delete"
                aria-label={`Delete conversation ${conversation.title}`}
                onClick={() => void handleDelete(conversation.id)}
              >
                🗑
              </button>
            </li>
          ))}
        </ul>
      </aside>
    </div>
  )
}
