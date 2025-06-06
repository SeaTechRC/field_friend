import logging
import os

import httpcore
import httpx
import rosys
from dotenv import load_dotenv
from nicegui import ui
from rosys.event import Event

load_dotenv('.env')

TELTONIKA_ROUTER_URL = 'http://192.168.42.1/api'
ADMIN_PASSWORD = os.environ.get('TELTONIKA_PASSWORD')


class TeltonikaRouter:
    def __init__(self) -> None:
        super().__init__()
        self.current_connection: str = 'disconnected'
        self.CONNECTION_CHANGED: Event = Event()

        self.log = logging.getLogger('hardware.teltonika_router')

        self.client = httpx.AsyncClient(headers={'Content-Type': 'application/json'}, timeout=20.0)
        self.auth_token: str = ''
        self.token_time: float = 0.0
        self.connection_check_running = False
        self.mobile_upload_permission = False
        self.MOBILE_UPLOAD_PERMISSION_CHANGED: Event = Event()
        if ADMIN_PASSWORD:
            self.log.info('Connecting to Teltonika router...')
            rosys.on_repeat(self.get_current_connection, 1.0)
        else:
            msg = 'The admin password for the Teltonika router is not set. Please set it in the .env file.'
            self.log.warning(msg)
            ui.label(msg).classes('text-xl')

    async def _ensure_connection(self) -> None:
        if self.connection_check_running is True:
            return True
        self.connection_check_running = True
        if rosys.time() - self.token_time > 4 * 60:
            await self._get_token()
        if self.auth_token == '':
            self.log.error('No authentication token found.')
            self.connection_check_running = False
            return False
        return True

    async def get_multi_ap_enabled(self) -> None:
        assert await self._ensure_connection()
        try:
            response = await self.client.get(f'{TELTONIKA_ROUTER_URL}/wireless/interfaces/status',
                                             headers={'Authorization': f'Bearer {self.auth_token}'})
            response.raise_for_status()
        except (httpx.RequestError, httpcore.ConnectError):
            self.log.error('Getting WIFI Status failed')
            self.connection_check_running = False
            return
        return response.json()['data'][1]['disabled'] == False

    async def get_multi_ap_networks(self):
        assert await self._ensure_connection()
        try:
            response = await self.client.get(f'{TELTONIKA_ROUTER_URL}/wireless/multi_ap/config',
                                             headers={'Authorization': f'Bearer {self.auth_token}'})
            response.raise_for_status()
        except (httpx.RequestError, httpcore.ConnectError):
            self.log.error('Getting Multi-AP Configs failed')
            self.connection_check_running = False
            return
        return response.json()['data']

    async def set_multi_ap_network(self, network_ssid):
        networks = await self.get_multi_ap_networks()
        for n in networks:
            n['enabled'] = '1' if n['ssid'] == network_ssid else '0'
        payload = {
            'data': networks
        }
        assert await self._ensure_connection()
        try:
            response = await self.client.put(f'{TELTONIKA_ROUTER_URL}/wireless/multi_ap/config',
                                             headers={'Authorization': f'Bearer {self.auth_token}'},
                                             json=payload)
            response.raise_for_status()
        except (httpx.RequestError, httpcore.ConnectError):
            self.log.error('Setting Multi-AP Configs failed')
            self.connection_check_running = False
            return
        return response.json()['data']

    async def set_multi_ap_enabled(self, enabled: bool) -> None:
        assert await self._ensure_connection()
        payload = {
            'data': [
                {
                    'enabled': '1' if enabled else '0',
                    'id': '1',
                    '.type': 'wifi-iface'
                }
            ]
        }
        try:
            response = await self.client.put(f'{TELTONIKA_ROUTER_URL}/wireless/interfaces/config',
                                             headers={'Authorization': f'Bearer {self.auth_token}'},
                                             json=payload)
            response.raise_for_status()
        except (httpx.RequestError, httpcore.ConnectError):
            self.log.error(f'Setting Multi-AP enabled to {enabled} failed')
            self.connection_check_running = False
            return

    async def get_current_connection(self) -> None:
        assert await self._ensure_connection()
        self.log.debug('Getting internet connection info...')
        try:
            response = await self.client.get(f'{TELTONIKA_ROUTER_URL}/failover/status',
                                             headers={'Authorization': f'Bearer {self.auth_token}'})
            response.raise_for_status()
        except (httpx.RequestError, httpcore.ConnectError):
            self.log.error('Getting Internet Connection Info failed')
            self.connection_check_running = False
            return
        self.log.debug('Getting Internet Connection Info: success')
        up_connection = 'disconnected'
        for key, value in response.json()['data'].items():
            if value.get('status') == 'online':
                up_connection = key
                break
        last_connection = self.current_connection
        if up_connection == 'wan':
            self.current_connection = 'ether'
        elif 'ifWan' in up_connection or 'wifi' in up_connection:
            self.current_connection = 'wifi'
        elif up_connection in ('mob1s1a1', 'mob1s2a1'):
            self.current_connection = 'mobile'
        else:
            self.current_connection = 'disconnected'
        if last_connection != self.current_connection:
            self.CONNECTION_CHANGED.emit()
        self.connection_check_running = False
        return

    async def _get_token(self) -> None:
        try:
            self.log.info('Getting authentication token for Teltonika router...')
            response = await self.client.post(f'{TELTONIKA_ROUTER_URL}/login',
                                              json={'username': 'admin', 'password': ADMIN_PASSWORD})
            response.raise_for_status()
        except (httpx.RequestError, httpx.ConnectError):
            self.log.exception('Teltonika router request failed.')
            self.auth_token = ''
            self.token_time = 0.0
            return
        try:
            self.auth_token = response.json()['data']['token']
        except KeyError:
            self.log.exception('No token in response.')
            self.auth_token = ''
            self.token_time = 0.0
            return
        self.token_time = rosys.time()
        self.log.info('Getting authentication token for Teltonika router: success')
