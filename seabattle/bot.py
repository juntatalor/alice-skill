# coding: utf-8

from __future__ import unicode_literals

import logging
import os

from telegram import ext as telegram_ext

from seabattle import dialog_manager as dm, session

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def bot_handler(bot, update):
    session_obj = session.get(update.message.chat_id)
    dm_obj = dm.DialogManager(session_obj)
    dmresponse = dm_obj.handle_message(update.message.text)
    bot.send_message(chat_id=update.message.chat_id, text=dmresponse.text)


def error_handler(bot, update, error):
    logger.error('Update "{0}" caused error "{1}"', update, error)


REQUEST_KWARGS = {}
proxy_url = os.environ.get('PROXY_URL')
if proxy_url:
    REQUEST_KWARGS['proxy_url'] = 'socks5://18.130.106.205:9999'

updater = telegram_ext.Updater(token=os.environ.get('TELEGRAM_TOKEN'), request_kwargs=REQUEST_KWARGS)
dispatcher = updater.dispatcher
dispatcher.add_handler(telegram_ext.MessageHandler(telegram_ext.Filters.text, bot_handler))
updater.start_polling()
updater.idle()
